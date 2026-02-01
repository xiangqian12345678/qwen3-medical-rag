"""多轮对话Agent：多轮医疗对话 + 规划式 RAG Agent"""
import logging
import time
from typing import List

from langchain_classic.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
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
    start_time = time.time()

    # 直接对检索结果进行压缩
    compressor = LLMChainExtractor.from_llm(llm)

    # 使用压缩器直接压缩检索到的文档
    compressed_docs = compressor.compress_documents(
        documents=docs,  # 传入检索到的文档
        query=query  # 传入查询
    )

    compressed_docs_list: List[Document] = list(compressed_docs)

    generate_time = time.time() - start_time
    logger.info(f"  filter_low_correction_content: {{'generate_time': {generate_time:.3f}, 'docs_count': {len(compressed_docs_list)}}}")

    return compressed_docs_list


# 2.压缩过滤-基于大模型过滤低相关文档
def filter_low_correction_doc_llm(query: str, docs: List[Document], llm: BaseChatModel) -> List[Document]:
    """
        压缩过滤-过滤低相关文档
    """
    start_time = time.time()

    # 直接对检索结果进行压缩
    _filter = LLMChainFilter.from_llm(llm)
    filtered_docs = _filter.compress_documents(docs, query)
    filter_docs_list: List[Document] = list(filtered_docs)

    generate_time = time.time() - start_time
    logger.info(f"  filter_low_correction_doc_llm: {{'generate_time': {generate_time:.3f}, 'docs_count': {len(filter_docs_list)}}}")

    return filter_docs_list


# 3.压缩-基于嵌入模型过滤低相关文档
def filter_low_correction_doc_embeddings(query: str, docs: List[Document],
                                         embeddings_model: Embeddings,
                                         low_correction_threshold: float = 0.66) -> List[Document]:
    """
        压缩过滤-过滤低相关文档
    """
    # DashScope embedding API 限制输入长度为 2048 字符
    # 需要截断过长的文档内容
    truncated_docs = []
    for doc in docs:
        truncated_content = doc.page_content[:2048] if len(doc.page_content) > 2048 else doc.page_content
        truncated_docs.append(Document(page_content=truncated_content, metadata=doc.metadata))

    embeddings_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=low_correction_threshold)
    filtered_docs = embeddings_filter.compress_documents(truncated_docs, query)
    filter_docs_list: List[Document] = list(filtered_docs)

    return filter_docs_list


# 4.冗余过滤-基于向量引擎
def filter_redundant_doc_embeddings(query: str, docs: List[Document],
                                    embeddings_model: Embeddings,
                                    redundant_threshold: float = 0.95) -> List[Document]:
    """
        冗余过滤-过滤高相关的重复文档，只保留一份
    """
    # DashScope embedding API 限制输入长度为 2048 字符
    # 需要截断过长的文档内容
    truncated_docs = []
    for doc in docs:
        truncated_content = doc.page_content[:2048] if len(doc.page_content) > 2048 else doc.page_content
        truncated_docs.append(Document(page_content=truncated_content, metadata=doc.metadata))

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model, similarity_threshold=redundant_threshold)
    filtered_docs = redundant_filter.transform_documents(truncated_docs, query=query)
    filter_docs_list: List[Document] = list(filtered_docs)

    return filter_docs_list
