"""搜索图：用于执行单个查询的RAG检索流程"""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from typing_extensions import TypedDict

from app_config import APPConfig
from recall.kgraph import create_kgraph_search_tool
from recall.kgraph.kg_utils import json_to_list_document as kg_json_to_list_document
from recall.milvus import create_db_search_tool
from recall.milvus.embed_utils import json_to_list_document
from recall.search import create_web_search_tool
from recall.search.search_utils import json_to_list_document as web_json_to_list_document

logger = logging.getLogger(__name__)


# =============================================================================
# 搜索图状态定义
# =============================================================================
class RecallState(TypedDict, total=False):
    query: str  # 用户查询
    other_messages: List  # 其他消息
    docs: List[Document]  # 检索到的文档


# =============================================================================
# 搜索工具
# =============================================================================
class RecallGraph:
    """搜索图：执行单个查询的RAG检索流程"""

    def __init__(self, app_config: APPConfig, llm: BaseChatModel, embed_model: Embeddings = None) -> None:
        """
        初始化搜索图

        Args:
            app_config: 应用配置
            llm: 强大模型实例
            websearch_func: 网络搜索函数
        """
        self.appConfig = app_config

        # 1.创建数据库检索工具
        db_tool, db_llm = create_db_search_tool(app_config.milvus_config_loader, llm, embed_model=embed_model)
        self.db_search_tool = db_tool
        self.db_search_llm = db_llm

        # 2.创建网络搜索工具
        if app_config.agent_config.network_search_enabled:
            web_tool, web_llm = create_web_search_tool(
                search_cnt=app_config.agent_config.network_search_cnt,
                power_model=llm
            )
            self.network_search_tool = web_tool
            self.network_search_llm = web_llm
        else:
            self.network_search_tool = None
            self.network_search_llm = None

        # 3.创建知识图谱搜索工具
        if app_config.agent_config.kgraph_search_enabled:
            kgraph_tool, kgraph_llm \
                = create_kgraph_search_tool(app_config.kgraph_config_loader, llm, embed_model=embed_model)
            self.kgraph_search_tool = kgraph_tool
            self.kgraph_search_llm = kgraph_llm
        else:
            self.kgraph_search_tool = None
            self.kgraph_search_llm = None

    def search(self, query: str) -> List[Document]:
        docs: List[Document] = []

        # 1. 调用数据库检索
        tool_result = self.db_search_tool.invoke({"query": query})
        db_docs = json_to_list_document(tool_result)
        docs.extend(db_docs)

        # 2. 调用网络检索
        if self.appConfig.agent_config.network_search_enabled and self.network_search_tool is not None:
            tool_result = self.network_search_tool.invoke(query)
            web_docs = web_json_to_list_document(tool_result)
            docs.extend(web_docs)

        # 3. 调用知识图谱检索
        if self.appConfig.agent_config.kgraph_search_enabled and self.kgraph_search_tool is not None:
            tool_result = self.kgraph_search_tool.invoke(query)
            kg_docs = kg_json_to_list_document(tool_result)
            docs.extend(kg_docs)

        return docs
