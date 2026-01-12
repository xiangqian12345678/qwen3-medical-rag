"""BM25稀疏向量处理"""
import logging
from typing import Dict, List

from langchain_core.embeddings import Embeddings

from .vocab import Vocabulary

logger = logging.getLogger(__name__)


class BM25Vectorizer:
    """BM25向量化器"""

    def __init__(
            self,
            vocab: Vocabulary,
            domain_model: str = "medicine",
            k1: float = 1.5,
            b: float = 0.75
    ):
        """
        初始化BM25向量化器

        Args:
            vocab: 词表
            domain_model: 领域模型
            k1: BM25参数k1
            b: BM25参数b
        """
        self.vocab = vocab
        self.domain_model = domain_model
        self.k1 = k1
        self.b = b

    def build_sparse_vector(self, text: str, avgdl: float = None) -> Dict[int, float]:
        """
        构建BM25稀疏向量

        Args:
            text: 输入文本
            avgdl: 平均文档长度，默认使用词表中的avgdl

        Returns:
            稀疏向量字典 {token_id: score}
        """
        if avgdl is None:
            avgdl = self.vocab.avgdl if self.vocab.avgdl > 0 else 100.0

        token_ids = self.vocab.tokenize(text, self.domain_model)
        if not token_ids:
            return {}

        # 计算词频
        tf = {}
        for token_id in token_ids:
            tf[token_id] = tf.get(token_id, 0) + 1

        # 计算文档长度
        doc_len = len(token_ids)

        # BM25公式
        sparse_vector = {}
        for token_id, freq in tf.items():
            if token_id not in self.vocab.idf:
                continue

            idf = self.vocab.idf[token_id]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
            score = idf * (numerator / denominator)

            sparse_vector[token_id] = score

        return sparse_vector


class BM25SparseEmbedding(Embeddings):
    """BM25稀疏向量嵌入（LangChain接口）"""

    def __init__(self, vocab: Vocabulary, vectorizer: BM25Vectorizer):
        """
        初始化

        Args:
            vocab: 词表
            vectorizer: BM25向量化器
        """
        self.vocab = vocab
        self.vectorizer = vectorizer

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        批量文档嵌入

        Args:
            texts: 文本列表

        Returns:
            稀疏向量列表
        """
        avgdl = self.vocab.avgdl if self.vocab.avgdl > 0 else 100.0
        return [self.vectorizer.build_sparse_vector(text, avgdl) for text in texts]

    def embed_query(self, text: str) -> Dict[int, float]:
        """
        查询嵌入

        Args:
            text: 查询文本

        Returns:
            稀疏向量
        """
        avgdl = self.vocab.avgdl if self.vocab.avgdl > 0 else 100.0
        return self.vectorizer.build_sparse_vector(text, avgdl)
