"""Milvus 向量检索模块"""
import json
import logging
from typing import List
from typing_extensions import TypedDict

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from .embed_loader import EmbedConfigLoader
from .embed_config import SingleSearchRequest
from .embed_searcher import get_kb
from .embed_templates import get_prompt_template
from .embed_utils import json_to_list_document, _should_call_tool


class DBRecallState(TypedDict, total=False):
    query: str
    other_messages: List
    docs: List[Document]


logger = logging.getLogger(__name__)


def llm_db_search(
        state: "DBRecallState",
        llm: BaseChatModel,
        db_tool_node: ToolNode,
        show_debug: bool
) -> "DBRecallState":
    """
    数据库检索节点

    ========== 功能说明 ==========
    该节点负责：
    1. 接收用户查询，让LLM判断是否需要调用数据库检索工具
    2. 如果需要，执行数据库检索并获取相关文档
    3. 将检索到的文档添加到状态中供后续RAG使用
    """
    print('-' * 60)
    print("开始向量检索")
    print('-' * 60)

    query = state["query"]

    if show_debug:
        logger.info(f"开始db检索节点，查询: {query}")

    # 调用LLM，让其判断是否需要调用数据库检索工具
    db_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("call_db")["system"]),
        HumanMessage(content=get_prompt_template("call_db")["user"].format(query=query))
    ])
    state["other_messages"].append(db_ai)

    # 检查LLM是否决定调用工具
    if _should_call_tool(db_ai):
        if show_debug:
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
            # 执行工具调用
            tool_msgs: ToolMessage = db_tool_node.invoke([db_ai])
            state["other_messages"].append(tool_msgs)

            # 检查工具返回内容
            tool_content = tool_msgs[0].content if tool_msgs else ""
            if show_debug:
                logger.info(f"工具返回内容长度: {len(tool_content)}")
                if tool_content:
                    logger.info(f"工具返回内容前200字符: {tool_content[:200]}")

            # 如果返回空内容,跳过JSON解析
            if not tool_content or not tool_content.strip():
                logger.warning("工具返回空内容,跳过JSON解析")
                return state

            # 将ToolMessage中的JSON字符串转换为Document对象列表
            new_docs = json_to_list_document(tool_content)
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


def create_db_search_tool(
        embed_config_loader: EmbedConfigLoader,
        power_model: BaseChatModel
):
    """
    创建数据库检索工具节点

    Args:
        embed_config_loader: 应用配置
        power_model: LLM实例

    Returns:
        tuple: (db_search_tool, db_search_llm, db_tool_node)
    """

    # 向量检索对应的collection_name
    fixed_collection_name = embed_config_loader.milvus.collection_name or embed_config_loader.default_search.collection_name

    def _create_requests(embed_config_loader: EmbedConfigLoader) -> List[SingleSearchRequest]:
        """
        根据配置动态生成多路检索请求

        Args:
            embed_config_loader: 配置加载器

        Returns:
            List[SingleSearchRequest]: 检索请求列表
        """
        requests = []

        # 遍历稠密向量字段配置，生成对应的检索请求
        for field_name, field_config in embed_config_loader.dense_fields.items():
            if field_config.embed:  # 只有启用了嵌入的字段才生成检索请求
                request = SingleSearchRequest(
                    anns_field=field_config.index_field,
                    limit=50,
                    search_params=field_config.search_params or {"ef": 64},
                    metric_type=field_config.metric_type
                )
                requests.append(request)

        # 遍历稀疏向量字段配置，生成对应的检索请求
        for field_name, field_config in embed_config_loader.sparse_fields.items():
            if field_config.embed:  # 只有启用了嵌入的字段才生成检索请求
                request = SingleSearchRequest(
                    anns_field=field_config.index_field,
                    limit=50,
                    search_params=field_config.search_params or {},
                    metric_type=field_config.metric_type
                )
                requests.append(request)

        # 如果没有配置任何检索字段，返回默认的三个检索请求
        if not requests:
            requests = [
                SingleSearchRequest(
                    anns_field="chunk_dense",
                    limit=50,
                    search_params={"ef": 64},
                    metric_type="COSINE"
                ),
                SingleSearchRequest(
                    anns_field="parent_chunk_dense",
                    limit=50,
                    search_params={"ef": 64},
                    metric_type="COSINE"
                ),
                SingleSearchRequest(
                    anns_field="questions_dense",
                    limit=50,
                    search_params={"ef": 64},
                    metric_type="COSINE"
                )
            ]

        return requests


    @tool("database_search")
    def database_search(query: str) -> str:
        """
        使用本地向量数据库进行检索
        Args: query: 检索查询文本
        Returns: 检索结果的JSON字符串
        """
        try:
            from .embed_config import SearchRequest, FusionSpec
            kb = get_kb(embed_config_loader.as_dict)

            # 动态生成检索请求
            search_requests = _create_requests(embed_config_loader)

            search_req = SearchRequest(
                query=query,
                collection_name=fixed_collection_name,
                requests=search_requests,
                fuse=FusionSpec(method=embed_config_loader.fusion.method, k=embed_config_loader.fusion.k),
                output_fields=embed_config_loader.default_search.output_fields,
                top_k=embed_config_loader.default_search.top_k,
                limit=embed_config_loader.default_search.limit
            )

            results = kb.search(req=search_req)
            logger.info(f"检索到 {len(results)} 条结果")

            # 转换Document对象为字典列表
            results_dict = []
            for doc in results:
                results_dict.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

            json_str = json.dumps(results_dict, ensure_ascii=False)
            logger.info(f"返回JSON字符串长度: {len(json_str)}")

            return json_str
        except Exception as e:
            logger.error(f"database_search 工具执行失败: {e}")
            # 返回空数组的JSON字符串
            return json.dumps([], ensure_ascii=False)

    db_search_tool = database_search
    db_search_llm = power_model.bind_tools([db_search_tool])
    db_tool_node = ToolNode([db_search_tool])

    return db_search_tool, db_search_llm, db_tool_node
