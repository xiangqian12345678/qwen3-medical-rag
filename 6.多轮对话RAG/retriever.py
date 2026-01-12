"""检索器 - LangChain标准接口"""
import logging
import time
from typing import Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config.models import SearchRequest
from knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class KnowledgeRetriever(BaseRetriever):
    """医疗检索器 - LangChain标准接口"""

    def __init__(self, knowledge_base: KnowledgeBase, search_config: SearchRequest):
        """
        初始化检索器

        Args:
            knowledge_base: 知识库实例
            search_config: 检索配置
        """
        super().__init__()
        object.__setattr__(self, 'knowledge_base', knowledge_base)
        object.__setattr__(self, 'search_config', search_config)

    def _get_relevant_documents(
            self,
            inputs: dict
    ) -> Dict[str, Any]:
        """
        Args:   inputs: 输入字典，包含"input"键（查询文本）
        Returns: 包含文档和检索时间的字典
        """
        query = inputs.get("input", "")
        self.search_config.query = query

        start_time = time.time()
        documents = self.knowledge_base.search(self.search_config)
        search_time = time.time() - start_time

        return {
            "documents": documents,
            "search_time": search_time
        }
