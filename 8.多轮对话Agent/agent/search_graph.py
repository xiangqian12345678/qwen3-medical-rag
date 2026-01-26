"""搜索图：用于执行单个查询的RAG检索流程"""
import logging
from functools import partial
from typing import List, Union

from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from config.models import AppConfig
from core.utils import create_llm_client
from ..prompts.templates import get_prompt_template
from .tools import AgentTools
from .utils import del_think, json_to_list_document, format_document_str, _should_call_tool

logger = logging.getLogger(__name__)


# =============================================================================
# 搜索图状态定义
# =============================================================================
class SearchMessagesState(TypedDict, total=False):
    """
    搜索消息状态

    用于在RAG检索流程中传递和管理状态信息。该状态类定义了整个搜索流程中
    所需的数据结构和流转信息，包括用户查询、对话历史、检索文档、生成答案等。

    ========== 使用案例 ==========
    场景：用户问"阿司匹林有哪些副作用？"

    初始状态：
    ```python
    init_state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [HumanMessage(content="阿司匹林有哪些副作用？")],
        "other_messages": [],
        "docs": [],
        "answer": "",
        "retry": 3,  # 最多重试3次
        "final": "",
        "judge_result": ""
    }
    ```

    流程演进：

    1. 数据库检索后（llm_db_search）：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [HumanMessage(content="阿司匹林有哪些副作用？")],
        "other_messages": [AIMessage(...), ToolMessage(...)],  # LLM调用和工具返回
        "docs": [
            Document(page_content="阿司匹林常见副作用包括：胃肠道反应、出血倾向、过敏反应等..."),
            Document(page_content="阿司匹林严重副作用：消化道溃疡、颅内出血等...")
        ],
        "answer": "",
        "retry": 3,
        "final": "",
        "judge_result": ""
    }
    ```

    2. RAG生成后（rag_node）：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [
            HumanMessage(content="阿司匹林有哪些副作用？"),
            AIMessage(content="根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...")
        ],
        "other_messages": [AIMessage(...), ToolMessage(...)],
        "docs": [Document(...), Document(...)],
        "answer": "根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...",  # RAG生成的答案
        "retry": 3,
        "final": "",
        "judge_result": ""
    }
    ```

    3. 质量判断后（judge_node）：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [...],
        "other_messages": [AIMessage(...), ToolMessage(...), AIMessage(content="[JUDGE]=yes")],
        "docs": [Document(...), Document(...)],
        "answer": "根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...",
        "retry": 3,
        "final": "",
        "judge_result": "pass"  # 质量判断通过
    }
    ```

    4. 成功结束后（finish_success）：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [...],
        "other_messages": [...],
        "docs": [Document(...), Document(...)],
        "answer": "根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...",
        "retry": 3,
        "final": "根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...",  # 最终答案
        "judge_result": "pass"
    }
    ```

    重试场景示例：

    如果judge判断为retry：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [...],
        "other_messages": [...],
        "docs": [Document(...), Document(...)],  # 可能会替换或补充文档
        "answer": "答案不完整...",
        "retry": 2,  # 剩余2次重试机会（初始3次-1）
        "final": "",
        "judge_result": "retry"
    }
    ```

    ========== 字段详解 ==========

    Attributes:
        query: 用户查询
            - 类型: str
            - 作用: 存储原始用户问题，贯穿整个搜索流程
            - 示例: "阿司匹林有哪些副作用？"

        main_messages: 主要对话消息
            - 类型: List[Union[HumanMessage, AIMessage]]
            - 作用: 存储用户和AI的主要对话历史，用于生成最终回答
            - 包含: HumanMessage（用户消息）和AIMessage（AI回答）
            - 示例: [HumanMessage(content="问题"), AIMessage(content="回答")]

        other_messages: 其他消息
            - 类型: List[Union[SystemMessage, ToolMessage]]
            - 作用: 存储系统消息和工具调用消息，用于调试和追踪
            - 包含: SystemMessage（系统提示）、ToolMessage（工具返回）
            - 示例: [AIMessage(content="工具调用决策"), ToolMessage(content="检索结果")]

        docs: 检索到的文档
            - 类型: List[Document]
            - 作用: 存储从数据库/网络检索到的相关文档
            - 结构: Document(page_content="内容", metadata={...})
            - 示例: [Document(page_content="药物说明..."), Document(page_content="临床数据...")]

        answer: RAG生成的回答
            - 类型: str
            - 作用: 存储RAG模块生成的原始答案（可能需要质量评估）
            - 示例: "根据检索到的文档，该药物的副作用包括..."

        retry: 剩余重试次数
            - 类型: int
            - 作用: 在答案质量不合格时，允许重新生成的次数
            - 默认值: 通常为3次
            - 变化: 每次重试减1，减至0时停止重试

        final: 最终回答
            - 类型: str
            - 作用: 经过质量评估后的最终答案，可直接返回给用户
            - 特点: 质量不达标时会附加警告信息
            - 示例: "根据医学资料..." 或 "...（内容可能不属实）"

        judge_result: RAG质量评判结果
            - 类型: str
            - 作用: 存储质量评估结果，用于路由决策
            - 可能值:
                - "pass": 质量达标，流程结束
                - "retry": 质量不达标，重新生成（retry > 0）
                - "fail": 质量不达标且无重试机会，流程结束
            - 示例: "pass" / "retry" / "fail"

    ========== 字段间关系 ==========

    数据流向：
    query -> db_search/web_search -> docs -> rag -> answer -> judge -> judge_result -> final

    消息分类：
    - main_messages: 生成最终答案所需的对话流
    - other_messages: 辅助信息，用于调试和日志记录

    重试机制：
    retry初始值（如3）-> 判断为retry时减1 -> 为0时转为fail

    ========== total=False 说明 ==========

    total=False表示该TypedDict的所有字段都是可选的，不强制要求每次都提供所有字段。
    这样允许在不同节点只更新需要修改的字段，而不需要每次都传递完整状态。
    """
    query: str  # 用户查询
    main_messages: List[Union[HumanMessage, AIMessage]]  # 主要对话消息
    other_messages: List[Union[SystemMessage, ToolMessage]]  # 其他消息
    docs: List[Document]  # 检索到的文档
    answer: str  # RAG生成的回答
    retry: int  # 剩余重试次数
    final: str  # 最终回答
    judge_result: str  # RAG质量评判结果


# =============================================================================
# 网络搜索结果模型
# =============================================================================
class NetworkSearchResult(BaseModel):
    """网络搜索结果类"""
    need_search: bool = Field(description="是否需要进行网络搜索")
    search_query: str = Field(description="网络搜索查询词", default="")
    remain_doc_index: List[int] = Field(description="保留的文档索引列表", default=[])


# =============================================================================
# 节点函数
# =============================================================================
def llm_db_search(
        state: SearchMessagesState,
        llm: BaseChatModel,
        db_tool_node: ToolNode,
        show_debug: bool
) -> SearchMessagesState:
    """
    数据库检索节点

    ========== 功能说明 ==========
    该节点负责：
    1. 接收用户查询，让LLM判断是否需要调用数据库检索工具
    2. 如果需要，执行数据库检索并获取相关文档
    3. 将检索到的文档添加到状态中供后续RAG使用

    ========== db_ai 输入示例 ==========

    输入消息列表（传入 llm.invoke 的参数）：
    ```python
    [
        SystemMessage(content="你是一个医疗助手，需要判断是否调用数据库检索工具..."),
        HumanMessage(content="阿司匹林有哪些副作用？")
    ]
    ```

    ========== db_ai 输出示例 ==========

    情况1：LLM决定调用工具（正常检索流程）
    ```python
    db_ai = AIMessage(
        content="我需要查询阿司匹林的副作用信息，让我检索数据库...",
        tool_calls=[
            ToolCall(
                id="call_abc123",
                name="database_search",
                args={
                    "query": "阿司匹林 副作用",
                    "top_k": 3
                }
            )
        ]
    )
    ```

    情况2：LLM决定不调用工具（无需检索）
    ```python
    db_ai = AIMessage(
        content="根据已有信息可以直接回答，无需数据库检索。",
        tool_calls=[]
    )
    ```

    ========== db_tool_node.invoke([db_ai]) 示例 ==========

    输入：包含工具调用的AIMessage列表
    ```python
    input_messages = [
        AIMessage(
            content="我需要查询阿司匹林的副作用...",
            tool_calls=[
                ToolCall(
                    id="call_abc123",
                    name="database_search",
                    args={"query": "阿司匹林 副作用", "top_k": 3}
                )
            ]
        )
    ]
    ```

    输出：ToolMessage列表（数据库检索结果）
    ```python
    tool_msgs = [
        ToolMessage(
            tool_call_id="call_abc123",
            content='[
                {
                    "page_content": "阿司匹林常见副作用包括：1.胃肠道反应（恶心、呕吐、胃痛）2.出血倾向 3.过敏反应（皮疹、哮喘）",
                    "metadata": {"source": "med_doc_001", "score": 0.95}
                },
                {
                    "page_content": "阿司匹林严重副作用：消化道溃疡、颅内出血、肝肾功能损害",
                    "metadata": {"source": "med_doc_002", "score": 0.88}
                },
                {
                    "page_content": "阿司匹林禁忌症：活动性溃疡、出血性疾病、哮喘患者慎用",
                    "metadata": {"source": "med_doc_003", "score": 0.82}
                }
            ]'
        )
    ]
    ```

    ========== 数据流示例 ==========

    初始状态：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [HumanMessage(content="阿司匹林有哪些副作用？")],
        "other_messages": [],
        "docs": [],
        "answer": "",
        "retry": 3,
        "final": "",
        "judge_result": ""
    }
    ```

    执行检索后：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [HumanMessage(content="阿司匹林有哪些副作用？")],
        "other_messages": [
            AIMessage(content="我需要查询...", tool_calls=[...]),  # db_ai
            ToolMessage(content='[{"page_content": "..."}]')        # tool_msgs
        ],
        "docs": [
            Document(page_content="阿司匹林常见副作用包括：...", metadata={...}),
            Document(page_content="阿司匹林严重副作用：...", metadata={...}),
            Document(page_content="阿司匹林禁忌症：...", metadata={...})
        ],
        "answer": "",
        "retry": 3,
        "final": "",
        "judge_result": ""
    }
    ```
    """
    query = state["query"]

    if show_debug:
        logger.info(f"开始db检索节点，查询: {query}")

    # 调用LLM，让其判断是否需要调用数据库检索工具
    # 输入：系统提示词 + 用户查询
    # 输出：db_ai，可能包含工具调用决策（tool_calls字段）
    db_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("call_db")["system"]),
        HumanMessage(content=get_prompt_template("call_db")["user"].format(query=query))
    ])
    state["other_messages"].append(db_ai)

    # 检查LLM是否决定调用工具
    if _should_call_tool(db_ai):
        if show_debug:
            # 提取工具调用参数用于调试日志
            tool_calls = getattr(db_ai, 'tool_calls', None)
            if tool_calls and len(tool_calls) > 0:
                try:
                    if hasattr(tool_calls[0], 'args'):
                        args = tool_calls[0].args
                    elif isinstance(tool_calls[0], dict):
                        args = tool_calls[0].get('args', {})
                    else:
                        args = {}
                    logger.info(f"开始db检索，检索参数：{args}")
                except Exception as e:
                    logger.error(f"获取工具参数失败: {e}")

        try:
            # 执行工具调用：将包含tool_calls的AIMessage传给ToolNode
            # ToolNode会解析tool_calls并调用对应的数据库检索工具
            # 返回：ToolMessage列表，包含数据库检索结果（JSON格式）
            tool_msgs: ToolMessage = db_tool_node.invoke([db_ai])
            state["other_messages"].append(tool_msgs)

            # 将ToolMessage中的JSON字符串转换为Document对象列表
            new_docs = json_to_list_document(tool_msgs[0].content)
            state["docs"].extend(new_docs)

            if show_debug:
                logger.info(f"检索到 {len(new_docs)} 条文档")
                if len(state["docs"]) >= 2:
                    logger.info(
                        f"部分示例（共{len(state['docs'])}条）：\n\n{state['docs'][0].page_content[:200]}...\n\n{state['docs'][1].page_content[:200]}..."
                    )
                elif len(state["docs"]) == 1:
                    logger.info(f"仅检索一条数据：\n\n{state['docs'][0].page_content[:200]}")
                else:
                    logger.warning("未检索到任何文档！")
        except Exception as e:
            logger.error(f"检索过程出错: {e}")

    return state


def llm_network_search(
        state: SearchMessagesState,
        judge_llm: BaseChatModel,
        network_search_llm: BaseChatModel,
        network_tool_node: ToolNode,
        show_debug: bool
) -> SearchMessagesState:
    """
    联网检索节点

    ========== 功能说明 ==========
    该节点负责：
    1. 基于已有检索文档，判断是否需要联网搜索补充信息
    2. 如果需要，生成搜索查询并执行网络检索
    3. 根据判断结果，决定保留或替换已有的本地数据库文档
    4. 将网络检索结果整合到文档列表中供后续RAG使用

    ========== judge_chain 执行流程示例 ==========

    步骤1：构建 judge_messages 模板
    ```python
    # 从 templates.py 加载 web_router 提示词模板
    # 假设 get_prompt_template("web_router")["system"] 内容为：
    # "请分析以下查询和已有文档，判断是否需要联网搜索。
    #  {format_instructions}"
    #
    # format_instructions 为：
    # "The output should be formatted as a JSON instance conforming to the JSON schema below.
    #  {{'need_search': '是否需要搜索', 'search_query': '搜索词', 'remain_doc_index': [1,2,3]}}"

    judge_messages = ChatPromptTemplate.from_messages([
        ("system", "... {format_instructions} ..."),
        ("human", "查询: {query}\n\n已有文档: {docs}")
    ])
    ```

    步骤2：执行 judge_chain.invoke()
    ```python
    # 输入参数
    invoke_input = {
        "query": "2024年最新的阿司匹林副作用研究",
        "docs": "文档1: 阿司匹林是常见解热镇痛药，副作用包括...\n文档2: 2020年的研究显示..."
    }

    # judge_chain 内部执行流程：
    # 1. judge_messages.format(invoke_input)
    #    生成完整的提示词：
    #    System: "请分析以下查询和已有文档，判断是否需要联网搜索...
    #            输出格式为JSON..."
    #    Human: "查询: 2024年最新的阿司匹林副作用研究
    #            已有文档: 文档1: 阿司匹林是常见解热镇痛药，副作用包括...
    #                     文档2: 2020年的研究显示..."

    # 2. judge_llm.invoke(messages)
    #    LLM返回（可能包含思考标签）：
    #    "<think>查询要求2024年最新研究，但现有文档只有2020年的信息，
    #           需要联网搜索最新资料。</think>
    #     {
    #       'need_search': true,
    #       'search_query': '阿司匹林 副作用 2024 最新研究',
    #       'remain_doc_index': [1]
    #     }"

    # 3. RunnableLambda(lambda x: del_think(x.content))
    #    移除 <think> 标签内容：
    #    "{
    #       'need_search': true,
    #       'search_query': '阿司匹林 副作用 2024 最新研究',
    #       'remain_doc_index': [1]
    #     }"

    # 4. fixing_parser.parse(json_string)
    #    解析为 Pydantic 对象：
    #    NetworkSearchResult(
    #        need_search=True,
    #        search_query='阿司匹林 副作用 2024 最新研究',
    #        remain_doc_index=[1]
    #    )

    # 最终输出 result
    result = NetworkSearchResult(
        need_search=True,
        search_query='阿司匹林 副作用 2024 最新研究',
        remain_doc_index=[1]  # 表示保留第1个文档（索引1），删除第2个文档
    )
    ```

    ========== 数据流示例 ==========
    场景：查询最新研究，现有文档过时
    ```python
    # 输入状态
    state = {
        "query": "2024年最新的阿司匹林副作用研究",
        "docs": [
            Document(page_content="阿司匹林基本药理作用..."),  # 文档1，index=1
            Document(page_content="2020年的研究显示...")       # 文档2，index=2，过时
        ],
        ...
    }

    # 判断结果
    result = NetworkSearchResult(
        need_search=True,
        search_query="阿司匹林 副作用 2024 最新研究",
        remain_doc_index=[1]  # 只保留文档1，删除文档2
    )

    # 执行网络检索后
    state = {
        "query": "2024年最新的阿司匹林副作用研究",
        "docs": [
            Document(page_content="阿司匹林基本药理作用..."),  # 保留的文档1
            Document(page_content="2024年Nature研究显示..."),  # 网络检索结果1
            Document(page_content="2024年JAMA研究指出...")    # 网络检索结果2
        ],
        ...
    }
    ```
    """
    if show_debug:
        logger.info(f"检查是否缺失资料需要网络搜索...")

    # 创建 Pydantic 输出解析器：将 LLM 输出的 JSON 字符串解析为 NetworkSearchResult 对象
    parser = PydanticOutputParser(pydantic_object=NetworkSearchResult)

    # 创建错误修复解析器：当 LLM 输出格式不正确时，自动调用 judge_llm 修复 JSON
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=judge_llm)

    # 获取格式指令，并将大括号转义（用于 .format() 避免冲突）
    # 原始: "{ 'key': 'value' }" -> 转义后: "{{ 'key': 'value' }}"
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    # 构建判断链的提示词模板：判断是否需要联网搜索
    judge_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("web_router")["system"].format(format_instructions=format_instructions)),
        ("human", get_prompt_template("web_router")["user"])
    ])

    # 构建调用链的提示词模板：执行网络检索
    calling_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("call_web")["system"]),
        ("human", get_prompt_template("call_web")["user"])
    ])

    # 构建判断链：提示词 -> LLM -> 删除思考标签 -> 解析为结构化对象
    # 执行顺序：
    # 1. judge_messages: 格式化提示词，插入 format_instructions 和用户输入
    # 2. judge_llm: 调用 LLM 生成判断结果（JSON格式）
    # 3. RunnableLambda(del_think): 移除 LLM 输出中的 <think> 标签内容
    # 4. fixing_parser: 解析 JSON 为 NetworkSearchResult 对象，格式错误时自动修复
    judge_chain = judge_messages | judge_llm | RunnableLambda(
        lambda x: del_think(x.content)) | fixing_parser

    try:
        # 执行判断链，分析是否需要联网搜索
        # 输入：用户查询 + 已检索到的文档
        # 输出：NetworkSearchResult 对象
        result: NetworkSearchResult = judge_chain.invoke({
            "query": state['query'],
            "docs": format_document_str(state.get('docs', []))
        })
        if show_debug:
            logger.info(
                f"判断结果: {'需要网络检索' if result.need_search else '不需要网络检索'}, 检索文本：{result.search_query}"
            )

        # 记录判断结果到状态中，用于调试追踪
        judge_ai_content = f"分析结果: {result.model_dump()}"
        judge_ai = AIMessage(content=judge_ai_content)
        state["other_messages"].append(judge_ai)

    except Exception as e:
        # JSON 解析失败时，使用默认值（不进行网络搜索）
        logger.error(f"JSON解析错误: {e}")
        result = NetworkSearchResult(need_search=False, search_query="", remain_doc_index=[])
        judge_ai = AIMessage(content=f"解析失败，使用默认值: {result.model_dump()}")
        state["other_messages"].append(judge_ai)

    # 如果需要联网搜索，且搜索词不为空
    if result.need_search and result.search_query.strip():
        # 执行网络检索链：提示词 -> network_search_llm（带有搜索工具绑定）
        # network_search_llm 会根据搜索词决定调用网络搜索工具
        search_chain = calling_messages | network_search_llm
        search_ai = search_chain.invoke({"search_query": result.search_query})
        state["other_messages"].append(search_ai)

        # 检查 LLM 是否决定调用搜索工具
        if _should_call_tool(search_ai):
            # 执行工具调用：调用网络搜索工具
            tool_msgs: ToolMessage = network_tool_node.invoke([search_ai])
            state["other_messages"].append(tool_msgs)

            # 根据 remain_doc_index 处理已有文档：
            # - 如果 remain_doc_index=[1,3]，则只保留第1、3个文档，删除第2、4个
            # - 如果 remain_doc_index=[]，则删除所有已有文档，只使用网络检索结果
            # - 索引从1开始，所以要减1转换为0-based索引
            remain_doc = result.remain_doc_index
            if remain_doc:
                valid_indices = [i - 1 for i in remain_doc if 0 < i <= len(state.get("docs", []))]
                state["docs"] = [state["docs"][i] for i in valid_indices]
            else:
                state["docs"] = []

            # 将网络检索结果（JSON格式）转换为 Document 对象并添加到文档列表
            state["docs"].extend(json_to_list_document(tool_msgs[0].content))
            if show_debug:
                logger.info(f"网络检索完毕")
    else:
        # 不需要联网搜索，直接使用已有文档
        if show_debug:
            logger.info(f"信息完整，无需网络搜索...")

    return state


def rag_node(
        state: SearchMessagesState,
        llm: BaseChatModel,
        show_debug: bool
) -> SearchMessagesState:
    """RAG生成节点"""
    if show_debug:
        logger.info(f"开始RAG...当前文档数量: {len(state.get('docs', []))}")

    sys = get_prompt_template("basic_rag")["system"]
    user = get_prompt_template("basic_rag")["user"]

    prompt = [
        SystemMessage(content=sys),
        HumanMessage(
            content=user.format(
                all_document_str=format_document_str(state.get("docs", [])),
                input=state["query"]
            )
        )
    ]

    rag_response = llm.invoke(prompt)

    # 调试：记录 LLM 原始输出
    if show_debug:
        logger.info(f"LLM原始输出长度: {len(rag_response.content)}")
        logger.info(f"LLM原始输出(前200字符): {rag_response.content[:200]}")

    rag_content = del_think(rag_response.content)

    # 调试：记录 del_think 后的输出
    if show_debug:
        logger.info(f"del_think后长度: {len(rag_content)}")
        logger.info(f"del_think后内容(前200字符): {rag_content[:200]}")

    rag_ai = AIMessage(content=rag_content)

    if show_debug:
        logger.info(f"RAG回答: {rag_ai.content}")

    if not isinstance(state["main_messages"][-1], AIMessage):
        state["main_messages"].append(rag_ai)
    else:
        state["main_messages"].pop()
        state["main_messages"].append(rag_ai)

    state["answer"] = rag_ai.content
    return state


def judge_node(
        state: SearchMessagesState,
        llm: BaseChatModel,
        show_debug: bool
) -> SearchMessagesState:
    """质量判断节点"""
    if show_debug:
        logger.info(f"开始评估...")

    judge_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("judge_rag")["system"]),
        HumanMessage(
            content=get_prompt_template("judge_rag")["user"].format(
                format_document_str=format_document_str(state.get('docs', [])),
                query=state['query'],
                summary=state.get('answer', '')
            )
        )
    ])

    result = del_think(judge_ai.content or "").strip().lower()

    if show_debug:
        logger.info(f"评估结果: {result[:20]}")

    state["other_messages"].append(AIMessage(content=f"[JUDGE]={result}"))

    if 'y' in result:
        state["judge_result"] = "pass"
    else:
        retries_left = int(state.get("retry", 0))
        if retries_left > 0:
            state["retry"] = retries_left - 1
            state["judge_result"] = "retry"
        else:
            state["judge_result"] = "fail"

    return state


def finish_success(state: SearchMessagesState) -> SearchMessagesState:
    """成功结束节点"""
    state["final"] = (state.get("answer", "") or "").strip() or "（空）"
    return state


def finish_fail(state: SearchMessagesState) -> SearchMessagesState:
    """失败结束节点"""
    base = (state.get("answer", "") or "").strip() or "（空）"
    state["final"] = base + "\n\n（内容可能不属实）"
    return state


# =============================================================================
# 搜索图类
# =============================================================================
class SearchGraph:
    """搜索图：执行单个查询的RAG检索流程"""

    def __init__(self, config: AppConfig, power_model: BaseChatModel, websearch_func=None) -> None:
        """
        初始化搜索图

        Args:
            config: 应用配置
            power_model: 强大模型实例（用于工具调用）
            websearch_func: 网络搜索函数（可选）
        """
        self.config = config

        self.agent_tools = AgentTools(config)
        if websearch_func and config.agent.network_search_enabled:
            self.agent_tools.register_websearch(websearch_func)

        # 只有在启用网络搜索时才创建网络搜索工具
        if config.agent.network_search_enabled:
            self.network_search_tool = self.agent_tools.make_web_search_tool()
            self.network_search_llm = power_model.bind_tools([self.network_search_tool])
            self.network_tool_node = ToolNode([self.network_search_tool])
        else:
            self.network_search_tool = None
            self.network_search_llm = None
            self.network_tool_node = None

        self.db_search_tool = self.agent_tools.make_database_search_tool()
        self.db_search_llm = power_model.bind_tools([self.db_search_tool])
        self.llm = create_llm_client(config.llm)
        self.db_tool_node = ToolNode([self.db_search_tool])
        self.search_graph = None

    def build_search_graph(self):
        """构建搜索图"""

        def judge_router(state: SearchMessagesState) -> str:
            """路由函数"""
            return state.get("judge_result", "fail")

        g = StateGraph(SearchMessagesState)

        db_search_node_func = partial(
            llm_db_search,
            llm=self.db_search_llm,
            db_tool_node=self.db_tool_node,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("db_search", db_search_node_func)

        # 只在启用网络搜索时添加web_search节点
        if self.config.agent.network_search_enabled:
            network_search_node_func = partial(
                llm_network_search,
                judge_llm=self.llm,
                network_search_llm=self.network_search_llm,
                network_tool_node=self.network_tool_node,
                show_debug=self.config.multi_dialogue_rag.console_debug
            )
            g.add_node("web_search", network_search_node_func)

        rag_node_func = partial(
            rag_node,
            llm=self.llm,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )

        g.add_node("rag", rag_node_func)
        g.add_node("finish_success", finish_success)
        g.add_node("finish_fail", finish_fail)

        if self.config.agent.mode == "analysis":
            judge_node_func = partial(
                judge_node,
                llm=self.llm,
                show_debug=self.config.multi_dialogue_rag.console_debug
            )
            g.add_node("judge", judge_node_func)

            if self.config.agent.network_search_enabled:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "web_search")
                g.add_edge("web_search", "rag")
            else:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "rag")

            g.add_edge("rag", "judge")
            g.add_conditional_edges(
                "judge",
                judge_router,
                {
                    "pass": "finish_success",
                    "retry": "rag",
                    "fail": "finish_fail"
                }
            )
            g.add_edge("finish_success", END)
            g.add_edge("finish_fail", END)

        elif self.config.agent.mode == "fast":
            if self.config.agent.network_search_enabled:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "web_search")
                g.add_edge("web_search", "rag")
            else:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "rag")
            g.add_edge("rag", END)

        else:  # normal模式
            if self.config.agent.network_search_enabled:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "web_search")
                g.add_edge("web_search", "rag")
            else:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "rag")
            g.add_edge("rag", END)

        self.search_graph = g.compile()

    def run(self, init_state: SearchMessagesState) -> SearchMessagesState:
        """
        执行搜索图
        Args:    init_state: 初始状态
        Returns: 执行完成后的最终状态
        """
        if self.search_graph is None:
            self.build_search_graph()

        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state

    def answer(self, query: str) -> str:
        """
        回答用户问题

        Args:
            query: 用户问题

        Returns:
            回答结果
        """
        if self.search_graph is None:
            self.build_search_graph()

        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "answer": "",
            "retry": self.config.agent.max_attempts,
            "final": "",
            "judge_result": ""
        }

        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state.get("final", "") or out_state.get("answer", "") or "（空）"
