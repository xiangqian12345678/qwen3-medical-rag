"""词汇表管理模块 - 用于构建和管理BM25词表"""

import gzip
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional


class Vocabulary:
    """维护 token->id 与 id->df，用于稀疏向量化"""

    def __init__(self):
        # token到ID的映射表，用于唯一标识每个token
        self.token_to_id: Dict[str, int] = {}

        # 文档频率统计，记录每个token出现在多少个文档中 (document frequency)
        # key: token_id, value: 包含该token的文档数
        self.doc_freq: Dict[int, int] = {}

        # 总文档数，用于计算IDF (inverse document frequency)
        self.total_docs: int = 0

        # 所有文档长度之和，用于计算平均文档长度
        self.sum_doc_length: int = 0

        # 预计算的IDF数组，冻结后生成，加速查询 (inverse document frequency array)
        self.idf_array: Optional[List[float]] = None

    def add_document(self, tokens: List[str]) -> None:
        """
        添加一个文档到词表中，统计词频和文档频率

        Args:
            tokens: 文档分词后的token列表
        """
        self.total_docs += 1
        self.sum_doc_length += len(tokens)

        # 记录当前文档中已出现的token，避免同一token在同一文档中重复计数df
        tokens_seen_in_doc = set()

        for token in tokens:
            # 如果token是新的，分配一个新的唯一ID
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

            # 获取token对应的ID
            token_id = self.token_to_id[token]

            # 如果该token在当前文档中首次出现，增加其文档频率
            if token_id not in tokens_seen_in_doc:
                self.doc_freq[token_id] = self.doc_freq.get(token_id, 0) + 1
                tokens_seen_in_doc.add(token_id)

        # 词表更新后，IDF数组失效，需要重新计算
        self.idf_array = None

    def idf(self, token_id: int) -> float:
        """
        计算 token 的逆文档频率 (IDF, Inverse Document Frequency)

        使用 BM25 公式: IDF = log(1 + (N - df + 0.5) / (df + 0.5))
        其中 N 是总文档数，df 是包含该token的文档数

        Args:
            token_id: token的唯一标识ID

        Returns:
            该token的IDF值，用于计算BM25分数
        """
        # 如果已预计算IDF数组，直接返回
        if self.idf_array is not None:
            return self.idf_array[token_id]

        # 动态计算IDF
        doc_freq = self.doc_freq.get(token_id, 0)
        return math.log(1.0 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def freeze(self) -> None:
        """
        冻结词表，预计算所有 IDF 值以加速查询

        冻结后，词表将预计算所有token的IDF值并存储在idf_array中，
        查询时可以直接返回，避免重复计算。
        """
        # 词表大小（唯一token的数量）
        vocab_size = len(self.token_to_id)

        # 构建文档频率数组，索引为token_id，值为对应的文档频率
        doc_freq_array = [0] * vocab_size
        for token_id, count in self.doc_freq.items():
            doc_freq_array[token_id] = count

        # 预计算所有token的IDF值
        self.idf_array = [
            math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
            for df in doc_freq_array
        ]

    def save(self, path: str, compress: bool = True) -> None:
        """
        保存词表到文件

        Args:
            path: 输出文件路径
            compress: 是否使用gzip压缩，默认为True
        """
        # 构建词表状态字典
        state = {
            "version": 1,  # 序列化版本号，用于未来兼容性处理
            "token2id": self.token_to_id,  # token到ID的映射
            "df": self.doc_freq,  # 文档频率统计
            "N": self.total_docs,  # 总文档数
            "sum_dl": self.sum_doc_length,  # 文档总长度
            "idf_arr": self.idf_array,  # 预计算的IDF数组
        }

        # 使用最高协议版本序列化
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

        # 确保输出目录存在
        output_dir = Path(path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 先写入临时文件，成功后再替换目标文件（原子性操作）
        tmp_path = path + ".tmp"
        with (gzip.open(tmp_path, "wb") if compress else open(tmp_path, "wb")) as f:
            f.write(data)

        # 原子性替换文件
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> Optional['Vocabulary']:
        """
        从文件加载词表

        Args:
            path: 词表文件路径，支持自动检测gzip压缩格式

        Returns:
            加载成功的Vocabulary实例，失败时返回None
        """
        try:
            # 根据文件扩展名自动判断是否使用gzip
            with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
                state = pickle.load(f)

            # 从状态字典重建词表对象
            vocab = cls()
            vocab.token_to_id = state["token2id"]
            vocab.doc_freq = state["df"]
            vocab.total_docs = state["N"]
            vocab.sum_doc_length = state.get("sum_dl", 0)
            vocab.idf_array = state.get("idf_arr", None)
            return vocab
        except Exception:
            # 加载失败时返回None，避免程序崩溃
            return None
