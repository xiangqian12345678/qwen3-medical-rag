import logging
from typing import List

import numpy as np
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

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

    # 创建查询-文档对
    pairs = [{ "query": doc.metadata["query"], "doc": doc} for doc in docs]

    # 计算相关性分数
    scores = []
    for pair in pairs:
        results = reranker.compress_documents(
            documents= [pair["doc"]],
            query=pair["query"]
        )
        scores.append(results[0].metadata["relevance_score"])

    # 按分数降序排序
    sorted_indices = np.argsort(scores)[::-1]

    # 获取排序后的文档和分数
    sorted_docs = [docs[i] for i in sorted_indices]

    return sorted_docs


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
