"""网络搜索模块"""
import json
import logging
from typing import List
from typing import TYPE_CHECKING

from langchain.output_parsers import OutputFixingParser
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from web_searcher import get_ws
from search_utils import del_think, format_document_str, json_to_list_document, _should_call_tool
from search_templates import get_prompt_template

if TYPE_CHECKING:
    from typing_extensions import TypedDict

    class SearchMessagesState(TypedDict, total=False):
        query: str
        main_messages: List
        other_messages: List
        docs: List[Document]
        answer: str
        retry: int
        final: str
        judge_result: str

logger = logging.getLogger(__name__)


# =============================================================================
# 网络搜索结果模型
# =============================================================================
class NetworkSearchResult(BaseModel):
    """网络搜索结果类"""
    need_search: bool = Field(description="是否需要进行网络搜索")
    search_query: str = Field(description="网络搜索查询词", default="")
    remain_doc_index: List[int] = Field(description="保留的文档索引列表", default=[])


def llm_network_search(
        state: "SearchMessagesState",
        judge_llm: BaseChatModel,
        network_search_llm: BaseChatModel,
        network_tool_node: ToolNode,
        show_debug: bool
) -> "SearchMessagesState":
    """
    联网检索节点

    ========== 功能说明 ==========
    该节点负责：
    1. 基于已有检索文档，判断是否需要联网搜索补充信息
    2. 如果需要，生成搜索查询并执行网络检索
    3. 根据判断结果，决定保留或替换已有的本地数据库文档
    4. 将网络检索结果整合到文档列表中供后续RAG使用
    """
    if show_debug:
        logger.info(f"检查是否缺失资料需要网络搜索...")

    # 创建 Pydantic 输出解析器
    parser = PydanticOutputParser(pydantic_object=NetworkSearchResult)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=judge_llm)

    # 获取格式指令，并将大括号转义
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    # 构建判断链的提示词模板
    judge_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("web_router")["system"].format(format_instructions=format_instructions)),
        ("human", get_prompt_template("web_router")["user"])
    ])

    # 构建调用链的提示词模板
    calling_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("call_web")["system"]),
        ("human", get_prompt_template("call_web")["user"])
    ])

    # 构建判断链
    judge_chain = judge_messages | judge_llm | RunnableLambda(
        lambda x: del_think(x.content)) | fixing_parser

    try:
        # 执行判断链
        result: NetworkSearchResult = judge_chain.invoke({
            "query": state['query'],
            "docs": format_document_str(state.get('docs', []))
        })
        if show_debug:
            logger.info(
                f"判断结果: {'需要网络检索' if result.need_search else '不需要网络检索'}, 检索文本：{result.search_query}"
            )

        # 记录判断结果
        judge_ai_content = f"分析结果: {result.model_dump()}"
        judge_ai = AIMessage(content=judge_ai_content)
        state["other_messages"].append(judge_ai)

    except Exception as e:
        # JSON 解析失败时，使用默认值
        logger.error(f"JSON解析错误: {e}")
        result = NetworkSearchResult(need_search=False, search_query="", remain_doc_index=[])
        judge_ai = AIMessage(content=f"解析失败，使用默认值: {result.model_dump()}")
        state["other_messages"].append(judge_ai)

    # 如果需要联网搜索，且搜索词不为空
    if result.need_search and result.search_query.strip():
        # 执行网络检索链
        search_chain = calling_messages | network_search_llm
        search_ai = search_chain.invoke({"search_query": result.search_query})
        state["other_messages"].append(search_ai)

        # 检查 LLM 是否决定调用搜索工具
        if _should_call_tool(search_ai):
            # 执行工具调用
            tool_msgs: ToolMessage = network_tool_node.invoke([search_ai])
            state["other_messages"].append(tool_msgs)

            # 根据 remain_doc_index 处理已有文档
            remain_doc = result.remain_doc_index
            if remain_doc:
                valid_indices = [i - 1 for i in remain_doc if 0 < i <= len(state.get("docs", []))]
                state["docs"] = [state["docs"][i] for i in valid_indices]
            else:
                state["docs"] = []

            # 将网络检索结果转换为 Document 对象并添加到文档列表
            try:
                tool_content = tool_msgs[0].content
                if tool_content and tool_content.strip():
                    state["docs"].extend(json_to_list_document(tool_content))
                    if show_debug:
                        logger.info(f"网络检索完毕，获取到 {len(tool_msgs[0].content.split())} 条结果")
                else:
                    if show_debug:
                        logger.warning(f"网络检索结果为空，跳过添加文档")
            except Exception as e:
                logger.error(f"解析网络检索结果失败: {e}, 工具内容: {tool_msgs[0].content[:200] if tool_msgs else 'No content'}")
                if show_debug:
                    logger.warning(f"网络检索结果解析失败，保留原有文档列表")
    else:
        # 不需要联网搜索
        if show_debug:
            logger.info(f"信息完整，无需网络搜索...")

    return state


def create_web_search_tool(
        config,
        power_model: BaseChatModel
):
    """
    创建网络搜索工具节点

    Args:
        config: 应用配置
        power_model: LLM实例

    Returns:
        tuple: (network_search_tool, network_search_llm, network_tool_node)
    """
    if config.agent.network_search_enabled is False:
        return None, None, None

    cnt = config.agent.network_search_cnt

    @tool("web_search")
    def web_search(query: str) -> str:
        """
        联网搜索工具

        Args:
            query: 搜索查询词

        Returns:
            检索结果的 JSON 字符串
        """
        web_searcher = get_ws(config.model_dump())

        results: List[Document] = web_searcher.search(query, cnt)
        # 转换Document对象为字典列表
        results_dict = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
        return json.dumps(
            results_dict,
            ensure_ascii=False,
            indent=2,
        )

    network_search_tool = web_search
    network_search_llm = power_model.bind_tools([network_search_tool])
    network_tool_node = ToolNode([network_search_tool])

    return network_search_tool, network_search_llm, network_tool_node
