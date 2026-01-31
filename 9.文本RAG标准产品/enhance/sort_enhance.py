import logging
import time
from typing import List

import numpy as np
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# 重排序
# 1.重排序-基于cross-encoder的重排序
def sort_docs_cross_encoder(docs: List[Document], reranker: DashScopeRerank) -> List[Document]:
    """
    重排序-基于cross-encoder的重排序

    Args:
        docs: 文档列表
        reranker: DashScopeRerank模型实例

    Returns:
        Tuple[排序后的文档列表, 对应的分数列表]
    """
    if not docs:
        return []

    # 按查询分组文档
    query_to_docs = {}
    for doc in docs:
        query = doc.metadata.get("query", "")
        if query not in query_to_docs:
            query_to_docs[query] = []
        query_to_docs[query].append(doc)

    # 对每个查询的文档进行批量重排序（带重试）
    all_reranked_docs = []
    for query, query_docs in query_to_docs.items():
        reranked_docs = _compress_documents_with_retry(
            reranker=reranker,
            documents=query_docs,
            query=query,
            max_retries=3,
            retry_delay=1.0
        )
        all_reranked_docs.extend(reranked_docs)

    # 如果所有文档的查询相同，使用API返回的排序结果
    # 否则按文档原始查询对应的分数进行排序
    if len(query_to_docs) == 1:
        return all_reranked_docs
    else:
        # 多个查询的情况，按分数排序
        scores = [doc.metadata.get("relevance_score", 0) for doc in all_reranked_docs]
        sorted_indices = np.argsort(scores)[::-1]
        sorted_docs = [all_reranked_docs[i] for i in sorted_indices]
        return sorted_docs


def _compress_documents_with_retry(
    reranker: DashScopeRerank,
    documents: List[Document],
    query: str,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Document]:
    """
    带重试机制的文档压缩重排序

    Args:
        reranker: DashScopeRerank模型实例
        documents: 文档列表
        query: 查询文本
        max_retries: 最大重试次数
        retry_delay: 重试延迟(秒)

    Returns:
        重排序后的文档列表
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"重排序文档数量: {len(documents)}, 尝试: {attempt + 1}/{max_retries}")
            results = reranker.compress_documents(
                documents=documents,
                query=query
            )
            logger.info(f"重排序成功，返回 {len(results)} 条文档")
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"重排序失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                logger.error(f"重排序失败，已达最大重试次数 {max_retries}: {e}")
                # 返回原始文档列表
                return documents


# 2.重排序-关键数据存储到首尾
def sort_docs_by_loss_of_location(docs: List[Document]) -> List[Document]:
    """
    重排序-关键数据存储到首尾

    Args:
        docs: 文档列表

    Returns:
        Tuple[排序后的文档列表, 对应的分数列表]
    """
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    reordered_docs_list: List[Document] = list(reordered_docs)

    return reordered_docs_list
