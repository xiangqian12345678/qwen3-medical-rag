"""多轮对话Agent：多轮医疗对话 + 规划式 RAG Agent"""
import logging
from typing import List

from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


# 过滤增强
# 1.压缩过滤-过滤低相关内容
def filter_low_correction_content(query: str, docs: List[Document], llm: BaseChatModel) -> List[Document]:
    """
        压缩过滤-过滤无关内容
    """
    # 直接对检索结果进行压缩
    compressor = LLMChainExtractor.from_llm(llm)

    # 使用压缩器直接压缩检索到的文档
    compressed_docs = compressor.compress_documents(
        documents=docs,  # 传入检索到的文档
        query=query  # 传入查询
    )

    compressed_docs_list: List[Document] = list(compressed_docs)

    return compressed_docs_list


# 2.压缩过滤-基于大模型过滤低相关文档
def filter_low_correction_doc_llm(query: str, docs: List[Document], llm: BaseChatModel) -> List[Document]:
    """
        压缩过滤-过滤低相关文档
    """
    # 直接对检索结果进行压缩
    _filter = LLMChainFilter.from_llm(llm)
    filtered_docs = _filter.compress_documents(docs, query)
    filter_docs_list: List[Document] = list(filtered_docs)

    return filter_docs_list


# 3.压缩-基于嵌入模型过滤低相关文档
def filter_low_correction_doc_embeddings(query: str, docs: List[Document],
                                         embeddings_model: Embeddings,
                                         low_correction_threshold: float = 0.66) -> List[Document]:
    """
        压缩过滤-过滤低相关文档
    """
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=low_correction_threshold)
    filtered_docs = embeddings_filter.compress_documents(docs, query)
    filter_docs_list: List[Document] = list(filtered_docs)

    return filter_docs_list


# 4.冗余过滤-基于向量引擎
def filter_redundant_doc_embeddings(query: str, docs: List[Document],
                                    embeddings_model: Embeddings,
                                    redundant_threshold: float = 0.95) -> List[Document]:
    """
        冗余过滤-过滤高相关的重复文档，只保留一份
    """
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model, similarity_threshold=redundant_threshold)
    filtered_docs = redundant_filter.transform_documents(docs, query=query)
    filter_docs_list: List[Document] = list(filtered_docs)

    return filter_docs_list
