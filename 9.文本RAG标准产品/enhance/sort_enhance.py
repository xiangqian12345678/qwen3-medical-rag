import logging
from typing import List, Tuple

import numpy as np
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# 重排序
# 1.重排序-基于cross-encoder的重排序
def sort_docs_cross_encoder(docs: List[Document], reranker: CrossEncoder) -> List[Document]:
    """
    重排序-基于cross-encoder的重排序

    Args:
        docs: 文档列表
        reranker: CrossEncoder模型实例

    Returns:
        Tuple[排序后的文档列表, 对应的分数列表]
    """
    if not docs:
        return []

    # 创建查询-文档对
    pairs = [[doc.metadata["query"], doc.document.page_content] for doc in docs]

    # 计算相关性分数
    scores = reranker.predict(pairs)

    # 按分数降序排序
    sorted_indices = np.argsort(scores)[::-1]

    # 获取排序后的文档和分数
    sorted_docs = [docs[i] for i in sorted_indices]

    return sorted_docs


def sort_docs_cross_encoder_v2(query: str, docs: List[Document], reranker: CrossEncoder) -> Tuple[
    List[Document], List[float]]:
    """
    重排序-基于cross-encoder的重排序，返回文档和分数

    Args:
        query: 查询字符串
        docs: 文档列表
        reranker: CrossEncoder模型实例

    Returns:
        Tuple[排序后的文档列表, 对应的分数列表]
    """
    if not docs:
        return [], []

    # 提取文档的文本内容
    doc_contents = [doc.page_content for doc in docs]

    # 创建查询-文档对
    pairs = [[query, doc_content] for doc_content in doc_contents]

    # 计算相关性分数
    scores = reranker.predict(pairs)

    # 按分数降序排序
    sorted_indices = np.argsort(scores)[::-1]

    # 获取排序后的文档和分数
    sorted_docs = [docs[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    return sorted_docs, sorted_scores


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
