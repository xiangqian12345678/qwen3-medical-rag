"""
知识图谱检索模块
参考milvus模块的实现，提供图谱检索工具
"""
import hashlib
import json
import logging
from typing import List

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .kg_loader import KGraphConfigLoader
from .kg_templates import get_prompt_template
from .kg_utils import json_to_list_document, _should_call_tool
from .kgraph_searcher import GraphSearcher
from .neo4j_connection import Neo4jConnection


class KGraphRecallState(TypedDict, total=False):
    query: str
    other_messages: List
    docs: List[Document]


logger = logging.getLogger(__name__)


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


def llm_kgraph_search(
        state: "KGraphRecallState",
        llm: BaseChatModel,
        kgraph_tool_node: ToolNode,
        show_debug: bool
) -> "KGraphRecallState":
    """
    知识图谱检索节点

    ========== 功能说明 ==========
    该节点负责：
    1. 接收用户查询，让LLM判断是否需要调用知识图谱检索工具
    2. 如果需要，执行图谱检索并获取相关实体和关系
    3. 将检索到的实体/关系转换为Document对象添加到状态中供后续RAG使用
    """
    logging.info('-' * 60)
    logging.info("开始图谱检索")
    logging.info('-' * 60)
    query = state["query"]

    if show_debug:
        logger.info(f"开始图谱检索节点，查询: {query}")

    # 调用LLM，让其判断是否需要调用图谱检索工具
    kg_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("call_kgraph")["system"]),
        HumanMessage(content=get_prompt_template("call_kgraph")["user"].format(query=query))
    ])
    state["other_messages"].append(kg_ai)

    # 检查LLM是否决定调用工具
    if _should_call_tool(kg_ai):
        if show_debug:
            tool_calls = getattr(kg_ai, 'tool_calls', None)
            if tool_calls and len(tool_calls) > 0:
                try:
                    if hasattr(tool_calls[0], 'args'):
                        args = tool_calls[0].args
                    elif isinstance(tool_calls[0], dict):
                        args = tool_calls[0].get('args', {})
                    else:
                        args = {}
                    logger.info(f"开始图谱检索，检索参数：{args}")
                except Exception as e:
                    logger.error(f"获取工具参数失败: {e}")

        try:
            # 执行工具调用
            tool_msgs: ToolMessage = kgraph_tool_node.invoke([kg_ai])
            state["other_messages"].append(tool_msgs)

            # 将ToolMessage中的JSON字符串转换为Document对象列表
            new_docs = json_to_list_document(tool_msgs[0].content)
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
    cnt = 10  # 默认检索10条结果
    neo4j_conn = Neo4jConnection(kgraph_config_loader)  # 创建Neo4j连接
    connected = neo4j_conn.connect()

    if not connected:
        logger.warning(f"Neo4j连接失败: {neo4j_conn.uri}")
        return None, None, None

    # 创建图谱检索器（传入嵌入配置以支持向量检索）
    # 使用 text_dense 配置作为嵌入模型
    embedding_config = {
        "provider": kgraph_config_loader.get("embedding.provider", "ollama"),
        "model": kgraph_config_loader.get("embedding.model", "nomic-embed-text"),
        "api_key": kgraph_config_loader.get("embedding.api_key", None),
        "base_url": kgraph_config_loader.get("embedding.base_url", "http://localhost:11434/v1")
    }
    graph_searcher = GraphSearcher(neo4j_conn, database=kgraph_config_loader.neo4j_config.database,
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
        results = graph_searcher.search_graph_by_query(query_text=query, top_k=cnt)
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
