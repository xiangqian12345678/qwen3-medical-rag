"""RAG基类"""
import logging
from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config.models import AppConfig, SearchRequest, SingleSearchRequest, FusionSpec

logger = logging.getLogger(__name__)


class BasicRAG(ABC):
    """基础医疗RAG系统"""

    def __init__(self, config: AppConfig, search_config: SearchRequest = None):
        self.config = config

        if not search_config:
            # 默认检索配置 - 使用 chunk_dense 和 chunk_sparse 混合检索
            from config.models import SingleSearchRequest, FusionSpec

            ssr1 = SingleSearchRequest(
                anns_field="chunk_dense",
                metric_type="COSINE",
                search_params={"ef": 64},
                limit=50,
                expr=""
            )
            ssr2 = SingleSearchRequest(
                anns_field="chunk_sparse",
                metric_type="IP",
                search_params={"drop_ratio_search": 0.0},
                limit=50,
                expr=""
            )
            fuse = FusionSpec(
                method="weighted",
                weights=[0.8, 0.2]
            )
            self.search_config = SearchRequest(
                query="",
                collection_name=config.milvus.collection_name,
                requests=[ssr1, ssr2],
                output_fields=["chunk", "parent_chunk", "summary", "questions", "document",
                             "source", "source_name", "lt_doc_id", "chunk_id", "hash_id"],
                fuse=fuse,
                top_k=50,
                limit=5
            )
        else:
            self.search_config = search_config

        logger.info("完成检索配置初始化")

    @abstractmethod
    def _setup_dialogue_rag_prompt(self) -> ChatPromptTemplate:
        """设置提示模板"""
        pass

    @abstractmethod
    def _setup_chain(self):
        """构建RAG检索链"""
        pass

    @abstractmethod
    def answer(
        self, query: str, return_document: bool = False
    ) -> Dict[str, Union[str, List[Document]]]:
        """
        Args:
            query: 用户问题
            return_document: 是否返回检索到的文档

        Returns:
            Dict({
                "answer": "...",
                "documents": [Document(..), Document(..)..]
            })
        """
        pass

    def batch_answer(
        self,
        queries: List[str],
        return_document: bool = False
    ) -> List[Dict[str, Union[str, List[Document]]]]:
        """批量回答问题"""
        results = []

        for query in queries:
            result = self.answer(query, return_document=return_document)
            results.append(result)

        return results

    @abstractmethod
    def update_search_config(self, search_config: SearchRequest):
        """更新检索配置并重建链"""
        pass
