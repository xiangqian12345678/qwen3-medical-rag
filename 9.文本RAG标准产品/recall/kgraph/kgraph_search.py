"""
知识图谱检索模块
参考milvus模块的实现，提供图谱检索工具
"""
import hashlib
import json
import logging
from typing import List, Optional

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool
from typing_extensions import TypedDict

from .kg_loader import KGraphConfigLoader
from .kg_utils import json_to_list_document
from .kgraph_searcher import GraphSearcher
from .neo4j_connection import Neo4jConnection

logger = logging.getLogger(__name__)
_gs_instance: Optional["GraphSearcher"] = None


class KGraphRecallState(TypedDict, total=False):
    query: str
    other_messages: List
    docs: List[Document]


def kgraph_search(
        state: "KGraphRecallState",
        kgraph_tool: Tool,
        show_debug: bool
) -> "KGraphRecallState":
    """
    知识图谱检索节点（直接调用工具，不使用LLM判断）

    ========== 功能说明 ==========
    该节点负责：
    1. 直接调用知识图谱检索工具进行查询
    2. 将检索到的实体/关系转换为Document对象添加到状态中供后续RAG使用

    Args:
        state: 包含查询和文档列表的状态
        kgraph_tool: 知识图谱检索工具
        show_debug: 是否显示调试信息

    Returns:
        更新后的状态，包含检索到的文档
    """
    logging.info('-' * 60)
    logging.info("开始图谱检索")
    logging.info('-' * 60)
    query = state["query"]

    if show_debug:
        logger.info(f"开始图谱检索节点，查询: {query}")

    try:
        # 直接调用工具函数进行检索
        result_json = kgraph_tool.func(query)

        # 将工具返回的JSON字符串转换为Document对象列表
        new_docs = json_to_list_document(result_json)
        state["docs"].extend(new_docs)

        if show_debug:
            logger.info(f"图谱检索到 {len(new_docs)} 条文档")
            if len(state["docs"]) >= 2:
                logger.info(
                    f"部分示例（共{len(state['docs'])}条）：\n\n{state['docs'][0].page_content[:200]}...\n\n{state['docs'][1].page_content[:200]}..."
                )
            elif len(state["docs"]) == 1:
                logger.info(f"仅检索一条数据：\n\n{state['docs'][0].page_content[:200]}")
            else:
                logger.warning("未检索到任何图谱信息！")
    except Exception as e:
        logger.error(f"图谱检索过程出错: {e}")

    return state


def create_kgraph_search_tool(
        kgraph_config_loader: KGraphConfigLoader,
        power_model: BaseChatModel,
        embed_model: Embeddings
):
    """
    创建知识图谱检索工具节点

    Args:
        kgraph_config_loader: 图谱配置加载器
        power_model: LLM实例
        embed_model: 嵌入模型

    Returns:
        tuple: (kgraph_search_tool, kgraph_search_llm, kgraph_tool_node)
    """
    # 默认启用知识图谱搜索
    cnt = kgraph_config_loader.neo4j_config.max_recall_num
    neo4j_conn = Neo4jConnection(kgraph_config_loader)  # 创建Neo4j连接
    connected = neo4j_conn.connect()

    if not connected:
        logger.warning(f"Neo4j连接失败: {neo4j_conn.uri}")
        return None, None, None

    graph_searcher = get_graph_searcher(neo4j_conn, database=kgraph_config_loader.neo4j_config.database,
                                        embed_model=embed_model)

    @tool("kgraph_search")
    def kgraph_search(query: str) -> str:
        """
        知识图谱检索工具

        Args:
            query: 检索查询文本

        Returns:
            检索结果的JSON字符串
        """
        # 使用向量检索
        results = graph_searcher.search_graph_by_query(query_text=query, power_model=power_model,top_k=cnt)
        vdb_results = results.get("vdb_results", [])

        # 转换为Document对象
        results_dict = [
            {
                "page_content": doc,
                "metadata": {
                    "source": "knowledge_graph",
                    "query": query,
                    "id": hashlib.md5(doc.encode('utf-8')).hexdigest()
                }
            }
            for doc in vdb_results
        ]

        return json.dumps(results_dict, ensure_ascii=False)

    kgraph_search_tool = kgraph_search
    kgraph_search_llm = power_model.bind_tools([kgraph_search_tool])

    return kgraph_search_tool, kgraph_search_llm


def get_graph_searcher(neo4j_conn: Neo4jConnection, database: str, embed_model: Embeddings) -> "GraphSearcher":
    global _gs_instance
    if _gs_instance is None:
        graph_searcher = GraphSearcher(neo4j_conn, database=database, embed_model=embed_model)
        _gs_instance = graph_searcher
    return _gs_instance
