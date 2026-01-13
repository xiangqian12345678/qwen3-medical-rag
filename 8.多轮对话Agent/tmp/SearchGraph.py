import json
import logging
import re
from copy import deepcopy
from functools import partial
from typing import List, Union
from typing import TypedDict

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from MedicalRag.agent.tools import AgentTools
from MedicalRag.prompts.templates import get_prompt_template
from .tools import tencent_cloud_search
from ..config.models import AppConfig
from ..core.utils import create_llm_client

logger = logging.getLogger(__name__)


def del_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def json_to_list_document(text):
    """
    将JSON格式的文本转换为Document对象列表

    该函数用于解析工具调用返回的JSON格式检索结果，将其转换为LangChain的Document对象
    常用于DB检索工具和网络检索工具返回结果的解析

    Args:
        text: JSON格式的字符串，包含文档数据的数组
              例如：'[{"page_content": "文档内容1", "metadata": {"source": "来源1"}}, 
                      {"page_content": "文档内容2", "metadata": {"source": "来源2"}}]'

    Returns:
        List[Document]: Document对象列表，每个Document包含：
            - page_content: 文档内容（str类型）
            - metadata: 元数据字典（dict类型），可能包含来源、分数等信息

    工作原理:
        1. 使用 json.loads() 将JSON字符串解析为Python列表
        2. 使用列表推导式遍历列表中的每个字典
        3. 通过解包操作符(**d)将字典的键值对作为参数传递给Document构造函数
           相当于：Document(page_content=d["page_content"], metadata=d["metadata"], ...)

    数据格式要求:
        JSON必须是数组格式，每个元素是包含Document所需字段的字典
        标准Document字段：
        - page_content (必需): 文档的文本内容
        - metadata (可选): 元数据字典，可以包含任意键值对

    使用场景:
        - 解析数据库检索工具的返回结果（向量检索、BM25检索）
        - 解析网络搜索工具的返回结果
        - 任何返回JSON格式文档数据的工具调用

    示例:
        # 输入JSON
        json_text = '''[
            {
                "page_content": "高血压是一种常见的心血管疾病...",
                "metadata": {
                    "source": "医学指南",
                    "score": 0.95
                }
            },
            {
                "page_content": "高血压的并发症包括中风、心脏病...",
                "metadata": {
                    "source": "医学文献",
                    "score": 0.88
                }
            }
        ]'''

        # 调用函数
        documents = json_to_list_document(json_text)

        # 输出结果
        # documents[0].page_content = "高血压是一种常见的心血管疾病..."
        # documents[0].metadata = {"source": "医学指南", "score": 0.95}

    异常处理:
        - 如果text不是有效的JSON格式，json.loads()会抛出JSONDecodeError
        - 如果JSON解析后的元素缺少Document必需的字段，Document构造函数会抛出ValidationError
        - 调用者应确保传入的JSON格式正确

    注意:
        - 该函数不包含异常处理，调用者需要处理可能的异常
        - 解包操作(**d)会将字典中的所有键值对传递给Document构造函数
        - 如果字典包含Document不支持的额外字段，可能会导致错误
    """
    return [Document(**d) for d in json.loads(text)]


def format_document_str(documents: List[Document]) -> str:
    parts = []
    for i, d in enumerate(reversed(documents)):
        if i >= 6:  # 做一个简单的截断
            break
        parts.append(f"## 文档{i + 1}：\n{d.page_content}\n")
    return "".join(parts)


class SearchMessagesState(TypedDict, total=False):
    """
    搜索消息状态类：定义搜索图执行过程中各节点间传递的状态数据结构

    该类继承自 TypedDict，提供类型提示但不强制运行时类型检查
    total=False 表示所有字段都是可选的（默认情况下）

    用途：
    - 作为 LangGraph StateGraph 的状态类型
    - 在各个节点（DB检索、网络检索、RAG生成、质量判断）之间传递和共享数据
    - 支持条件路由和状态判断

    字段说明：
        query: 用户查询字符串，作为整个RAG流程的输入
        main_messages: 主要对话消息列表，包含HumanMessage和AIMessage
                      用于记录主要的用户-模型交互历史
        other_messages: 其他辅助消息列表，包含SystemMessage、ToolMessage等
                       用于记录工具调用、系统提示、判断结果等辅助信息
        docs: 检索到的文档列表，由Document对象组成
              包含page_content（文档内容）和metadata（元数据，如来源、分数等）
        summary: RAG生成的摘要/答案字符串，存储LLM基于检索文档生成的回答
        retry: 剩余可重试次数，用于质量判断失败时的重试机制
               初始值为配置中的max_attempts，每次重试后递减
        final: 最终输出字符串，存储整个搜索图执行完成后的最终结果
               在fast模式下等于summary，在analysis模式下可能包含警告信息
        judge_result: 质量判断结果字符串，由judge节点设置
                      可能值：'pass'（通过）、'retry'（重试）、'fail'（失败）
                      用于条件路由，决定下一个执行节点
    """
    query: str  # 用户查询字符串，RAG流程的输入
    main_messages: List[Union[HumanMessage, AIMessage]]  # 主要对话消息：用户消息+AI消息
    other_messages: List[BaseMessage]  # 辅助消息：系统消息、工具消息、判断结果等
    docs: List[Document]  # 检索到的文档列表，包含文档内容和元数据
    summary: str  # RAG生成的摘要/答案，LLM基于检索文档生成的回答
    retry: int  # 剩余可重试次数，用于质量判断失败时的重试
    final: str  # 最终输出结果，整个搜索图执行完成后的答案
    judge_result: str  # 质量判断结果：'pass'/'retry'/'fail'，用于条件路由


class NetworkSearchResult(BaseModel):
    """
    网络搜索结果类：定义网络搜索判断节点的输出结构

    该类继承自 Pydantic BaseModel，用于结构化解析LLM的输出
    配合 PydanticOutputParser 使用，可以强制LLM输出符合结构的数据

    用途：
    - 在 llm_network_search 函数中，让LLM判断是否需要进行网络搜索
    - 通过 PydanticOutputParser 自动解析LLM输出为结构化对象
    - 支持容错解析（OutputFixingParser），自动修复格式错误

    字段说明：
        need_search: 布尔值，表示是否需要进行网络搜索
                    True表示DB检索结果不足，需要联网补充信息
                    False表示DB检索结果已足够满足查询需求
        search_query: 字符串，用于网络搜索的查询词
                     当need_search=True时，LLM会生成优化的搜索查询词
                     当need_search=False时，该字段为空字符串（default=""）
        remain_doc_index: 整数列表，表示需要保留的DB文档索引
                         索引从1开始（符合人类直觉，如[1,3]表示保留第1和第3个文档）
                         当需要进行网络搜索时，可以选择保留部分DB检索文档
                         空列表表示清空所有DB文档，完全使用网络搜索结果

    使用示例：
        # 判断需要搜索，保留第1个文档
        result = NetworkSearchResult(need_search=True, search_query="高血压最新治疗指南", remain_doc_index=[1])

        # 判断不需要搜索
        result = NetworkSearchResult(need_search=False, search_query="", remain_doc_index=[])

        # 需要搜索但清空所有DB文档
        result = NetworkSearchResult(need_search=True, search_query="新冠变异株症状", remain_doc_index=[])
    """
    need_search: bool = Field(description="是否需要进行网络搜索")  # True:需要联网 False:不需要
    search_query: str = Field(description="网络搜索查询词", default="")  # 用于网络搜索的查询关键词
    remain_doc_index: List[int] = Field(description="保留的文档索引列表", default=[])  # 需要保留的DB文档索引，从1开始


def _should_call_tool(last_ai: BaseMessage) -> bool:
    """
    判断上一步LLM输出是否触发了工具调用

    该函数用于检查LLM是否决定调用工具，而不是生成普通文本响应

    Args:
        last_ai: 上一步LLM生成的消息对象，类型为 BaseMessage
                可能是 AIMessage（AI响应消息）或其他消息类型

    Returns:
        bool: 如果消息包含工具调用则返回 True，否则返回 False

    工作原理：
        1. 使用 getattr 动态获取 last_ai 对象的 tool_calls 属性
        2. 如果该属性不存在，则返回默认值 None
        3. 将获取到的值转换为布尔值：
           - tool_calls 非空列表/元组 → True
           - tool_calls 为 None 或空列表 → False

    使用场景：
        - 在 llm_db_search 函数中：判断DB检索是否需要执行
        - 在 llm_network_search 函数中：判断网络搜索是否需要执行
        - 在任何需要区分"工具调用"和"普通响应"的场景

    注意：
        - 函数名前缀下划线(_)表示这是一个内部辅助函数
        - 该函数是工具调用机制的核心判断逻辑
        - LLM工具调用机制是 LangChain 的核心功能之一

    示例：
        # LLM决定调用工具
        msg = AIMessage(content="", tool_calls=[{"name": "search", "args": {...}}])
        _should_call_tool(msg)  # 返回 True

        # LLM生成普通响应
        msg = AIMessage(content="这是一个普通回答")
        _should_call_tool(msg)  # 返回 False
    """
    return bool(getattr(last_ai, "tool_calls", None))


def llm_db_search(
        state: SearchMessagesState,
        llm: BaseChatModel,
        db_tool_node: ToolNode,
        show_debug: bool
) -> SearchMessagesState:
    """
    数据库检索节点：负责从向量数据库中检索相关文档

    该节点是搜索图的第一个节点，通过LLM判断是否需要从数据库检索相关信息
    如果LLM决定调用工具，则执行向量检索和BM25检索，获取相关文档

    Args:
        state: 搜索消息状态对象，包含以下关键字段：
            - query: 用户查询字符串
            - other_messages: 辅助消息列表，用于记录LLM响应和工具调用结果
            - docs: 检索到的文档列表（List[Document]）
        llm: 绑定了数据库搜索工具的LLM实例
             该LLM可以调用db_search_tool进行向量检索
             通常使用能力较强的模型（如GPT-4）
        db_tool_node: 数据库工具节点（ToolNode对象）
                     负责执行数据库搜索工具并处理返回结果
        show_debug: 布尔值，是否输出调试日志
                   True时会在logger中输出检索参数和文档示例

    Returns:
        SearchMessagesState: 更新后的状态对象，包含以下变化：
            - other_messages: 追加db_ai（LLM响应）和tool_msgs（工具执行结果）
            - docs: 追加新检索到的文档列表（如果有）

    执行流程:
        1. 从状态中提取用户查询
        2. 构建提示词，包含系统提示和用户提示
           系统提示：指导LLM判断何时调用数据库检索工具
           用户提示：包含用户的原始查询问题
        3. 调用LLM进行推理，判断是否需要检索
        4. 将LLM响应保存到other_messages中
        5. 检查LLM是否触发了工具调用（通过_should_call_tool函数）
        6. 如果需要检索：
           - 输出调试日志（检索参数）
           - 执行工具节点，获取检索结果
           - 将工具消息保存到other_messages
           - 将检索到的JSON文档解析为Document对象列表
           - 将文档追加到state["docs"]中
           - 输出调试日志（文档示例）
        7. 如果不需要检索：
           - 直接返回原始状态，docs列表保持不变
        8. 返回更新后的状态

    工作原理:
        - LLM通过系统提示理解何时调用数据库检索工具
        - 工具调用通过LangChain的tool_calls机制实现
        - 检索结果以JSON格式返回，通过json_to_list_document函数解析
        - 支持多次调用时累积检索结果（使用extend追加）

    使用场景:
        - 用户问询医疗知识问题（如"高血压的并发症有哪些"）
        - 需要从医学知识库中检索相关文献和指南
        - 作为RAG流程的第一步，为后续的答案生成提供知识支撑

    示例:
        # 调用示例（在SearchGraph类中）
        db_search_node = partial(
            llm_db_search,
            llm=self.db_search_llm,  # 绑定了DB工具的LLM
            db_tool_node=self.db_tool_node,  # DB工具节点
            show_debug=True  # 开启调试
        )
        g.add_node("db_search", db_search_node)

    注意:
        - 该函数使用partial进行部分参数绑定，方便在StateGraph中使用
        - 检索到的文档会被追加到现有的docs列表中，不会覆盖
        - 调试日志会截断文档内容到前200个字符
        - 如果LLM判断不需要检索，则不会调用工具，直接跳过
    """
    query = state["query"]
    db_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("call_db")["system"]),
        HumanMessage(content=get_prompt_template("call_db")["user"].format(query=query))
    ])
    state["other_messages"].append(db_ai)
    if _should_call_tool(db_ai):
        if show_debug:
            logger.info(f"开始db检索，检索参数：{db_ai.additional_kwargs['tool_calls'][0]['function']['arguments']}")
        tool_msgs: ToolMessage = db_tool_node.invoke([db_ai])
        state["other_messages"].append(tool_msgs)
        state["docs"].extend(json_to_list_document(tool_msgs[0].content))
        if show_debug:
            if len(state["docs"]) >= 2:
                logger.info(
                    f"部分示例（共{len(state['docs'])}条）：\n\n{state['docs'][0].page_content[:200]}...\n\n{state['docs'][1].page_content[:200]}...")
            else:
                logger.info(f"仅检索一条数据：\n\n{state['docs'][0].page_content[:200]}")
    return state


def llm_network_search(
        state: SearchMessagesState,
        judge_llm: BaseChatModel,
        network_search_llm: BaseChatModel,
        network_tool_node: ToolNode,
        show_debug: bool
) -> SearchMessagesState:
    """
    联网检索节点：负责判断是否需要网络搜索，并执行网络检索（如需要）

    该节点在DB检索之后运行，通过LLM分析已检索的文档是否足以回答用户查询
    如果文档不足，则生成搜索查询词进行网络搜索，获取额外信息

    Args:
        state: 搜索消息状态对象，包含以下关键字段：
            - query: 用户查询字符串
            - docs: 已检索到的DB文档列表（List[Document]）
            - other_messages: 辅助消息列表，用于记录判断和搜索过程
        judge_llm: 判断LLM实例，用于判断是否需要网络搜索
                   不绑定任何工具，专注于结构化输出
        network_search_llm: 绑定了网络搜索工具的LLM实例
                            可以调用web_search_tool进行联网检索
                            通常使用能力较强的模型
        network_tool_node: 网络搜索工具节点（ToolNode对象）
                          负责执行网络搜索工具并处理返回结果
        show_debug: 布尔值，是否输出调试日志
                   True时会在logger中输出判断结果和检索信息

    Returns:
        SearchMessagesState: 更新后的状态对象，包含以下变化：
            - other_messages: 追加判断消息（judge_ai）和工具执行结果（tool_msgs）
            - docs: 根据remain_doc_index保留部分DB文档，并追加网络搜索结果（如果有）

    执行流程:
        步骤1：初始化解析器和提示词模板
        1.1 创建Pydantic解析器（parser），用于解析LLM输出为NetworkSearchResult对象
        1.2 创建容错解析器（fixing_parser），能自动修复LLM输出的格式错误
        1.3 获取格式指令并转义大括号（用于模板字符串）
        1.4 构建判断消息模板（judge_messages），包含系统提示和用户提示
        1.5 构建工具调用消息模板（calling_messages），用于实际的网络搜索

        步骤2：判断是否需要搜索
        2.1 构建判断链：judge_messages → judge_llm → 清理思考过程 → fixing_parser
        2.2 执行判断链，传入用户查询和已检索的文档
        2.3 获取判断结果（NetworkSearchResult对象）
        2.4 保存判断消息到状态（用于调试和追踪）

        步骤3：异常处理
        如果判断过程中出现JSON解析错误，使用默认值（不搜索）继续执行

        步骤4：执行网络搜索（如果需要）
        4.1 检查result.need_search和search_query是否有效
        4.2 构建搜索链：calling_messages → network_search_llm
        4.3 执行搜索链，传入优化的搜索查询词
        4.4 保存搜索消息到状态

        步骤5：处理搜索结果
        5.1 检查LLM是否触发了工具调用
        5.2 执行工具节点，获取网络搜索结果
        5.3 根据remain_doc_index保留或清空DB文档
        5.4 解析JSON格式的搜索结果，追加到docs列表

        步骤6：返回更新后的状态

    工作原理:
        - 使用PydanticOutputParser强制LLM输出结构化数据
        - 使用OutputFixingParser自动修复格式错误，提高鲁棒性
        - 使用RunnableLambda链式处理：提示词 → LLM → 清理 → 解析
        - 网络搜索通过LangChain工具调用机制实现
        - 支持选择性保留DB文档，避免完全替换已有信息

    判断逻辑:
        LLM根据以下信息判断是否需要网络搜索：
        - 用户查询（query）：了解用户的真实需求
        - DB检索文档（docs）：评估已有信息的完整性
        输出结构化判断：
        - need_search: 是否需要搜索
        - search_query: 优化的搜索查询词（如需要）
        - remain_doc_index: 需要保留的DB文档索引（如搜索）

    文档更新策略:
        - remain_doc_index非空：保留指定索引的文档，其余删除
        - remain_doc_index为空：清空所有DB文档，完全使用网络搜索结果
        - 索引从1开始（符合人类直觉），内部处理时转换为0开始

    使用场景:
        - DB检索结果不足，需要获取最新信息（如"2024年高血压治疗指南"）
        - 问题超出本地知识库范围（如"最新的新冠疫苗信息"）
        - 需要补充实时数据（如"当前流感病毒株情况"）

    示例:
        # 调用示例（在SearchGraph类中）
        network_search_node = partial(
            llm_network_search,
            judge_llm=self.llm,  # 不绑定工具的LLM
            network_search_llm=self.network_search_llm,  # 绑定web工具的LLM
            network_tool_node=self.network_tool_node,  # web工具节点
            show_debug=True  # 开启调试
        )
        g.add_node("web_search", network_search_node)

    注意:
        - judge_llm不应绑定任何工具，专注于结构化输出
        - network_search_llm必须绑定网络搜索工具
        - 大括号转义是必须的，否则会与Python的format语法冲突
        - 索引过滤使用列表推导式，避免越界错误
        - 异常处理使用默认值确保流程继续执行
        - 该函数使用partial进行部分参数绑定，方便在StateGraph中使用
    """
    if show_debug:
        logger.info(f"检查是否缺失资料需要网络搜索...")
    # 创建Pydantic解析器
    parser = PydanticOutputParser(pydantic_object=NetworkSearchResult)
    # 可选：创建容错解析器，能自动修复格式错误
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=judge_llm)  # 使用不绑定工具的LLM

    # 获取格式指令并转义大括号
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    # 构建判断消息模板
    judge_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("web_router")["system"].format(format_instructions=format_instructions)),
        ("human", get_prompt_template("web_router")["user"])
    ])

    # 构建工具调用消息模板
    calling_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("call_web")["system"]),
        ("human", get_prompt_template("call_web")["user"])
    ])

    # 步骤1：判断是否需要搜索
    judge_chain = judge_messages | judge_llm | RunnableLambda(
        lambda x: del_think(x.content)) | fixing_parser  # 使用不绑定工具的LLM

    try:
        # 执行判断链，直接得到解析后的结果
        result: NetworkSearchResult = judge_chain.invoke({
            "query": state['query'],
            "docs": format_document_str(state.get('docs', []))
        })
        if show_debug:
            logger.info(
                f"判断结果: {'需要网络检索' if result.need_search else '不需要网络检索'}, 检索文本：{result.search_query}")

        # 保存判断消息到状态（用于调试）
        judge_ai_content = f"分析结果: {result.model_dump()}"
        judge_ai = AIMessage(content=judge_ai_content)
        state["other_messages"].append(judge_ai)

    except Exception as e:
        logger.error(f"JSON解析错误: {e}")
        # 默认值
        result = NetworkSearchResult(need_search=False, search_query="", remain_doc_index=[])
        judge_ai = AIMessage(content=f"解析失败，使用默认值: {result.model_dump()}")
        state["other_messages"].append(judge_ai)

    # 步骤2：如果需要搜索，执行工具调用
    if result.need_search and result.search_query.strip():

        # 创建搜索链
        search_chain = calling_messages | network_search_llm

        # 执行搜索
        search_ai = search_chain.invoke({"search_query": result.search_query})
        state["other_messages"].append(search_ai)

        # 检查是否有工具调用
        if _should_call_tool(search_ai):
            tool_msgs: ToolMessage = network_tool_node.invoke([search_ai])
            state["other_messages"].append(tool_msgs)

            # 更新文档
            remain_doc = result.remain_doc_index
            if remain_doc:
                # 过滤有效索引，避免越界
                valid_indices = [i - 1 for i in remain_doc if 0 < i <= len(state.get("docs", []))]
                state["docs"] = [state["docs"][i] for i in valid_indices]
            else:
                state["docs"] = []  # 如果没有指定保留文档，清空原文档

            # 添加新搜索到的文档
            state["docs"].extend(json_to_list_document(tool_msgs[0].content))
            if show_debug:
                logger.info(f"网络检索完毕")
    else:
        if show_debug:
            logger.info(f"信息完整，无需网络搜索...")

    return state


def rag(
        state: SearchMessagesState,
        llm: BaseChatModel,
        show_debug: bool
) -> SearchMessagesState:
    """
    RAG生成节点：基于检索到的文档生成最终回答

    输入状态（state）依赖字段：
        - state["query"]: 用户原始问题
        - state["docs"]: 检索到的 Document 列表
        - state["main_messages"]: 主对话消息历史

    输出 / 修改状态字段：
        - state["main_messages"]: 追加或替换 AIMessage（本轮RAG回答）
        - state["summary"]: 本轮生成的答案文本（str）
    """

    # ---------- 1. 调试日志 ----------
    if show_debug:
        logger.info(f"开始RAG...")

    # ---------- 2. 读取 RAG Prompt 模板 ----------
    # system: 约束模型角色、回答风格、是否引用文档等
    # user: 注入 documents + query 的主提示
    sys = get_prompt_template("basic_rag")["system"]
    user = get_prompt_template("basic_rag")["user"]

    # ---------- 3. 构造 RAG Prompt ----------
    # 将 docs 格式化为可读文本（最多截断）
    # 注入到 user prompt 中
    prompt = [
        SystemMessage(content=sys),
        HumanMessage(
            content=user.format(
                all_document_str=format_document_str(state.get("docs", [])),
                input=state["query"]
            )
        )
    ]

    # ---------- 4. 调用 LLM 生成回答 ----------
    # 返回的是 AIMessage
    rag_ai = llm.invoke(prompt)

    # ---------- 5. 清理模型的思考过程（<think>...</think>） ----------
    # 保证 summary 是干净的可展示文本
    rag_ai.content = del_think(rag_ai.content)

    # ---------- 6. 维护 main_messages（支持 retry 场景） ----------
    # 如果上一条不是 AIMessage：
    #   - 说明是第一次 RAG 生成，直接 append
    #
    # 如果上一条已经是 AIMessage：
    #   - 说明是 retry 产生的新回答
    #   - 删除上一轮不合格的回答，再写入新结果
    if not isinstance(state["main_messages"][-1], AIMessage):
        # 上一轮 rag 生成合格（第一次生成）
        state["main_messages"].append(rag_ai)
    else:
        # 上一轮 rag 生成不合格（retry 场景）
        state["main_messages"].pop()
        state["main_messages"].append(rag_ai)

    # ---------- 7. 写入 summary ----------
    # summary 是 judge / final 输出的核心依赖字段
    state["summary"] = rag_ai.content

    # ---------- 8. 返回更新后的状态 ----------
    return state


def judge(
        state: SearchMessagesState,
        llm: BaseChatModel,
        show_debug: bool
) -> SearchMessagesState:
    """
    RAG质量判断节点：评估当前 summary 是否满足要求

    输入状态依赖字段：
        - state["query"]: 用户问题
        - state["docs"]: 检索文档
        - state["summary"]: RAG生成的答案
        - state["retry"]: 剩余可重试次数

    输出 / 修改状态字段：
        - state["judge_result"]: pass / retry / fail
        - state["retry"]: 若 retry，递减
        - state["other_messages"]: 追加判断日志消息
    """

    # ---------- 1. 调试日志 ----------
    if show_debug:
        logger.info(f"开始评估...")

    # ---------- 2. 构造 Judge Prompt 并调用 LLM ----------
    # Judge Prompt 的核心任务：
    #   - 判断 summary 是否“可信 / 完整 / 基于文档”
    #   - 通常要求输出 yes / no 或类似结果
    judge_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("judge_rag")["system"]),
        HumanMessage(
            content=get_prompt_template("judge_rag")["user"].format(
                format_document_str=format_document_str(state.get('docs', [])),
                query=state['query'],
                summary=state.get('summary', '')
            )
        )
    ])

    # ---------- 3. 清理 judge 输出 ----------
    # 去除 <think>，统一小写，便于规则判断
    result = del_think(judge_ai.content or "").strip().lower()

    if show_debug:
        logger.info(f"评估结果{result[:20]}")

    # ---------- 4. 保存 judge 结果到 other_messages（仅做日志追踪） ----------
    state["other_messages"].append(
        AIMessage(content=f"[JUDGE]={result}")
    )

    # ---------- 5. 根据结果修改状态（核心路由逻辑） ----------
    # 判断逻辑非常简单：
    #   - 包含 'y' => 通过（yes）
    #   - 否则 => retry 或 fail
    if 'y' in result:
        # 评估通过，后续会走 finish_success
        state["judge_result"] = "pass"
    else:
        retries_left = int(state.get("retry", 0))

        if retries_left > 0:
            # 还有重试次数：
            #   - retry 次数 -1
            #   - 标记为 retry，Graph 会路由回 rag
            state["retry"] = retries_left - 1
            state["judge_result"] = "retry"
        else:
            # 没有重试次数了，直接失败
            state["judge_result"] = "fail"

    # ---------- 6. 返回更新后的状态 ----------
    return state


class SearchGraph:
    def __init__(self, config: AppConfig, power_model: BaseChatModel, websearch_func=tencent_cloud_search) -> None:
        """
        初始化搜索图（SearchGraph）实例

        Args:
            config: 应用配置对象，包含模型配置、RAG配置、Agent配置等
            power_model: 强大模型实例，用于绑定工具（DB搜索、网络搜索），通常使用能力较强的模型
            websearch_func: 网络搜索函数，默认使用腾讯云搜索（tencent_cloud_search）
        """
        # 保存应用配置，用于读取各类配置参数（如调试开关、Agent模式等）
        self.config = config

        # 初始化Agent工具集合，负责创建和管理各类工具
        self.agent_tools = AgentTools(self.config)

        # 注册网络搜索函数到工具集合中
        self.agent_tools.register_websearch(websearch_func)

        # 创建数据库搜索工具（向量检索、BM25检索等）
        self.db_search_tool = self.agent_tools.make_database_search_tool()

        # 创建网络搜索工具（联网检索）
        self.network_search_tool = self.agent_tools.make_web_search_tool()

        # 创建绑定数据库搜索工具的LLM（用于DB检索节点）
        # 使用deepcopy避免污染原始power_model实例
        self.db_search_llm = deepcopy(power_model).bind_tools([self.db_search_tool])

        # 创建绑定网络搜索工具的LLM（用于网络检索节点）
        self.network_search_llm = deepcopy(power_model).bind_tools([self.network_search_tool])

        # 创建普通LLM客户端（用于RAG生成、质量判断等）
        # 从配置中读取LLM配置
        self.llm = create_llm_client(self.config.llm)

        # 创建数据库工具节点（LangGraph ToolNode）
        # 负责执行DB搜索工具并处理返回结果
        self.db_tool_node = ToolNode([self.db_search_tool])

        # 创建网络搜索工具节点（LangGraph ToolNode）
        # 负责执行网络搜索工具并处理返回结果
        self.network_tool_node = ToolNode([self.network_search_tool])

        # 搜索图实例（稍后通过build_search_graph方法构建）
        self.search_graph = None

    def build_search_graph(self):
        """
        构建搜索图：创建一个包含数据库检索、网络检索、RAG生成、质量判断等节点的状态图
        支持两种模式：analysis（带质量判断和重试机制）和 fast（快速模式，无判断）
        """

        def judge_router(state: SearchMessagesState) -> str:
            """简单的路由函数：只读取状态中的judge_result，决定下一个节点"""
            return state.get("judge_result", "fail")

        def finish_success(state: SearchMessagesState) -> SearchMessagesState:
            """结束节点：成功输出，将summary作为final结果"""
            state["final"] = (state.get("summary", "") or "").strip() or "（空）"
            return state

        def finish_fail(state: SearchMessagesState) -> SearchMessagesState:
            """结束节点：失败警告输出，在summary后添加可能不属实的警告"""
            base = (state.get("summary", "") or "").strip() or "（空）"
            state["final"] = base + "\n\n（内容可能不属实）"
            return state

        # 创建状态图，使用SearchMessagesState作为状态类型
        g = StateGraph(SearchMessagesState)

        # 原子节点配置：使用partial函数预先绑定LLM和工具节点等参数
        # db_search_node: 数据库检索节点，负责向量检索和BM25检索
        # partial函数: 来自 Python 的 functools 模块，它的作用是部分参数绑定，
        #   固定一个函数的部分参数，返回一个新的可调用对象
        db_search_node = partial(
            llm_db_search,
            llm=self.db_search_llm,  # 绑定DB搜索工具的LLM
            db_tool_node=self.db_tool_node,  # DB工具节点
            show_debug=self.config.multi_dialogue_rag.console_debug  # 调试开关
        )
        g.add_node("db_search", db_search_node)

        # web_search_node: 网络检索节点，负责联网搜索获取额外信息
        network_search_node = partial(
            llm_network_search,
            judge_llm=self.llm,  # 用于判断是否需要搜索的LLM
            network_search_llm=self.network_search_llm,  # 绑定网络搜索工具的LLM
            network_tool_node=self.network_tool_node,  # 网络搜索工具节点
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("web_search", network_search_node)

        # rag_node: RAG生成节点，根据检索到的文档生成答案
        rag_node = partial(
            rag,
            llm=self.llm,  # 用于RAG生成的LLM
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("rag", rag_node)

        # 成功和失败结束节点
        g.add_node("finish_success", finish_success)
        g.add_node("finish_fail", finish_fail)

        # judge_node: 质量判断节点，评估RAG生成质量并决定是否重试
        judge_node = partial(
            judge,
            llm=self.llm,  # 用于质量评估的LLM
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("judge", judge_node)

        # 设置图入口：从db_search节点开始
        g.set_entry_point("db_search")

        # 构建节点间连接关系
        # db_search -> web_search -> rag (如果启用了网络搜索)
        if self.config.agent.network_search_enabled:
            g.add_edge("db_search", "web_search")
            g.add_edge("web_search", "rag")
        else:
            # db_search -> rag (不使用网络搜索)
            g.add_edge("db_search", "rag")

        # rag之后的路由逻辑
        # analysis模式：带质量判断和重试机制
        if self.config.agent.mode == "analysis":
            g.add_edge("rag", "judge")  # RAG生成后进入判断节点
            # 从judge节点出发的条件边，根据judge_router的返回值路由
            g.add_conditional_edges(
                "judge",  # 从判断节点出发
                judge_router,  # 纯路由函数，读取judge_result状态
                {
                    "pass": "finish_success",  # 通过则结束
                    "retry": "rag",  # 重试则返回rag节点重新生成
                    "fail": "finish_fail",  # 失败则结束并添加警告
                }
            )
            # 设置结束节点连接到END
            g.add_edge("finish_success", END)
            g.add_edge("finish_fail", END)
        # fast模式：快速模式，RAG生成后直接结束，无质量判断
        elif self.config.agent.mode == "fast":
            g.add_edge("rag", END)

        # 编译图生成可执行的搜索图
        self.search_graph = g.compile()

    # ---------- 对外：跑整张图，返回最终输出 ----------
    def answer(self, query: str) -> str:
        if self.search_graph is None:
            self.build_search_graph()
        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "summary": "",
            "retry": self.config.agent.max_attempts,
            "final": ""
        }
        # 执行图
        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state.get("final", "") or out_state.get("summary", "") or "（空）"

    def run(self, init_state: SearchMessagesState) -> SearchMessagesState:
        """
        执行搜索图：运行整个RAG流程并返回最终状态

        Args:
            init_state: 初始状态，必须包含query等必需字段
                       SearchMessagesState结构:
                       - query: 用户查询
                       - main_messages: 主要对话消息列表
                       - other_messages: 其他辅助消息列表
                       - docs: 检索到的文档列表
                       - summary: RAG生成的摘要
                       - retry: 剩余重试次数
                       - final: 最终输出结果
                       - judge_result: 质量判断结果

        Returns:
            SearchMessagesState: 执行完成后的最终状态，包含final字段（最终答案）

        执行流程:
        1. 检查搜索图是否已构建，未构建则调用build_search_graph()构建
        2. 使用invoke方法执行搜索图，传入初始状态
        3. 返回执行完成后的最终状态

        注意:
        - 该方法与answer()方法的区别：
          * run(): 直接接收SearchMessagesState对象，返回完整状态
          * answer(): 接收query字符串，内部构建初始状态，仅返回final结果
        - 搜索图内部会自动处理DB检索、网络检索（如启用）、RAG生成、质量判断等流程
        """
        if self.search_graph is None:
            self.build_search_graph()
        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state
