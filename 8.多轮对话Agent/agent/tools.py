"""Agent工具类"""
import json
import logging
from typing import List

from langchain.tools import tool
from langchain_core.documents import Document

from config.models import AppConfig, SearchRequest
from core.db_factory import get_kb

logger = logging.getLogger(__name__)


class AgentTools:
    """Agent工具集合"""

    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        self.WEBSEARCH_FUNC = None

    def register_websearch(self, func):
        """注册网络搜索函数"""
        self.WEBSEARCH_FUNC = func

    def make_database_search_tool(self):
        """构造数据库检索Tool"""
        fixed_collection_name = self.app_config.milvus.collection_name

        @tool("database_search")
        def database_search(query: str) -> str:
            """
            使用本地向量数据库进行检索

            Args:
                query: 检索查询文本

            Returns:
                检索结果的JSON字符串
            """
            kb = get_kb(self.app_config.model_dump())

            search_req = SearchRequest(
                query=query,
                collection_name=fixed_collection_name,
                limit=5
            )

            results = kb.search(req=search_req)
            return json.dumps(
                [d.model_dump() for d in results],
                ensure_ascii=False
            )

        return database_search

    def make_web_search_tool(self):
        """构造网络搜索Tool"""
        if self.WEBSEARCH_FUNC is None:
            raise ValueError("未注册网络检索工具")

        cnt = self.app_config.agent.network_search_cnt

        @tool("web_search")
        def web_search(query: str) -> str:
            """
            联网搜索工具

            Args:
                query: 搜索查询词

            Returns:
                检索结果的JSON字符串
            """
            results: List[Document] = self.WEBSEARCH_FUNC(query, cnt)
            return json.dumps(
                [d.model_dump() for d in results],
                ensure_ascii=False
            )

        return web_search
