"""嵌入模块 - 支持稀疏向量"""
from .vocab import Vocabulary
from .bm25 import BM25Vectorizer, BM25SparseEmbedding

__all__ = [
    "Vocabulary",
    "BM25Vectorizer",
    "BM25SparseEmbedding",
]
