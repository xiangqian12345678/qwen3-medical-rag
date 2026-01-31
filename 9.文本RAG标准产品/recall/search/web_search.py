"""网络搜索模块"""
from langchain_core.tools import Tool

"""网络搜索模块"""
import logging
from typing import List

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from .search_utils import json_to_list_document
from .web_searcher import get_ws


class WebSearchState(TypedDict, total=False):
    query: str
    other_messages: List
    docs: List[Document]


logger = logging.getLogger(__name__)


# =============================================================================
# 网络搜索结果模型
# =============================================================================
class NetworkSearchResult(BaseModel):
    """网络搜索结果类"""
    need_search: bool = Field(description="是否需要进行网络搜索")
    search_query: str = Field(description="网络搜索查询词", default="")
    remain_doc_index: List[int] = Field(description="保留的文档索引列表", default=[])


def network_search(
        state: "WebSearchState",
        search_tool: Tool,
        show_debug: bool = False
) -> "WebSearchState":
    """
    网络检索节点（直接调用，不使用大模型判断）

    ========== 功能说明 ==========
    该节点负责：
    1. 接收用户查询，直接调用网络搜索工具
    2. 执行网络检索并获取相关文档
    3. 将检索到的文档添加到状态中供后续RAG使用

    Args:
        state: WebSearchState状态
        search_tool: 网络搜索工具实例
        show_debug: 是否显示调试信息
    """
    logging.info('-' * 60)
    logging.info("开始网络检索")
    logging.info('-' * 60)

    query = state["query"]

    if show_debug:
        logger.info(f"开始网络检索节点，查询: {query}")

    try:
        # 直接调用搜索工具
        if show_debug:
            logger.info(f"直接调用搜索工具，查询: {query}")

        if search_tool:
            tool_result = search_tool.invoke(query)

            if show_debug:
                logger.info(f"工具返回内容长度: {len(tool_result)}")
                if tool_result:
                    logger.info(f"工具返回内容前200字符: {tool_result[:200]}")

            # 如果返回空内容，跳过JSON解析
            if not tool_result or not tool_result.strip():
                logger.warning("工具返回空内容，跳过JSON解析")
                return state

            # 将工具返回的JSON字符串转换为Document对象列表
            new_docs = json_to_list_document(tool_result)
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
        else:
            logger.error("search_tool 未提供，无法执行网络搜索")
    except Exception as e:
        logger.error(f"检索过程出错: {e}")

    return state


def create_web_search_tool(
        search_cnt: int = 10,
        power_model: BaseChatModel = None
):
    """
    创建网络搜索工具

    Args:
        search_cnt: 网络搜索返回结果数量
        power_model: LLM实例

    Returns:
        tuple: (web_search_tool, web_search_llm)
    """
    if power_model is None:
        return None, None

    # 提前获取 WebSearcher 实例
    web_searcher = get_ws()

    @tool("web_search")
    def web_search(query: str) -> str:
        """
        使用网络搜索引擎进行检索
        Args: query: 检索查询文本
        Returns: 检索结果的JSON字符串
        """
        results: List[Document] = web_searcher.search(query, search_cnt)
        # 转换Document对象为字典列表
        results_dict = []
        for doc in results:
            results_dict.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        # 将字典列表转换为JSON字符串
        import json
        json_str = json.dumps(results_dict, ensure_ascii=False)
        logger.info(f"返回JSON字符串长度: {len(json_str)}")

        return json_str

    web_search_tool = web_search
    web_search_llm = power_model.bind_tools([web_search_tool])

    return web_search_tool, web_search_llm
