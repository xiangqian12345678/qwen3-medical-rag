"""词表管理（用于稀疏向量）"""
import gzip
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache

current_dir = Path(__file__).resolve().parent
default_vocab_dir = str(current_dir) + "/vocab/"


@lru_cache()
def get_stopwords(source: str = "all"):
    """获取停用词表

    Args:
        source: 停用词来源

    Returns:
        set: 停用词集合
    """
    supported_source = ["cn", "baidu", "hit", "scu", "marimo", "ict", "iso", "all"]
    if source not in supported_source:
        raise NotImplementedError(f"未知来源: {source}")

    stopwords_dir = Path(__file__).resolve().parent.parent / "core" / "stopwords"
    stopwords_file = stopwords_dir / f"stopwords.zh.{source}.txt"

    if not stopwords_file.exists():
        raise FileNotFoundError(f"停用词文件不存在: {stopwords_file}")

    return set(stopwords_file.read_text(encoding='utf8').strip().split())


def filter_stopwords(tokens: List[str], source: str = "all") -> List[str]:
    """过滤停用词

    Args:
        tokens: token列表
        source: 停用词来源

    Returns:
        List[str]: 过滤后的token列表
    """
    stopwords_set = get_stopwords(source)
    return [t for t in tokens if t not in stopwords_set]


class Vocabulary:
    """维护 token->id 与 id->df，用于稀疏向量化"""

    def __init__(self):
        # token -> id 的映射字典，用于将每个词映射到一个唯一的整数ID
        # 例如: {'感冒': 0, '发热': 1, '咳嗽': 2}
        self.token2id: Dict[str, int] = {}
        # document frequency (df)，即包含该token的文档数量
        # key: token_id, value: 包含该token的文档数
        # 例如: {0: 150, 1: 200, 2: 120} 表示token_id=0的词出现在150个文档中
        self.df: Dict[int, int] = {}
        # 文档总数，记录已处理的文档数量
        # 用于计算 IDF (Inverse Document Frequency)
        self.N: int = 0
        # 所有文档的token总长度之和，即 Σ(len(document_i))
        # 用于计算平均文档长度 (avgdl = sum_dl / N)
        self.sum_dl: int = 0
        # IDF数组，缓存所有token的逆文档频率值
        # 建库结束后一次性计算并缓存，避免频繁计算
        # index与token_id对应: idf_arr[token_id] = 该token的idf值
        self.idf_arr: Optional[List[float]] = None

    def add_document(self, tokens: List[str]):
        """添加一个文档到词表

        Args:
            tokens: 分词后的token列表
        """
        self.N += 1
        self.sum_dl += len(tokens)
        seen = set()
        for t in tokens:
            if t not in self.token2id:
                self.token2id[t] = len(self.token2id)
            tid = self.token2id[t]
            if tid not in seen:
                self.df[tid] = self.df.get(tid, 0) + 1
                seen.add(tid)
        self.idf_arr = None

    def idf(self, tid: int) -> float:
        """计算 token 的逆文档频率

        Args:
            tid: token id

        Returns:
            float: idf值
        """
        if self.idf_arr is not None:
            return self.idf_arr[tid]
        df = self.df.get(tid, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    @property
    def avgdl(self) -> float:
        """平均文档长度"""
        return self.sum_dl / self.N if self.N > 0 else 0

    def freeze(self):
        """建库结束后一次性缓存 idf"""
        V = len(self.token2id)
        df_arr = [0] * V
        for tid, c in self.df.items():
            df_arr[tid] = c
        self.idf_arr = [math.log(1.0 + (self.N - d + 0.5) / (d + 0.5)) for d in df_arr]

    def save(self, path: str, compress: bool = True):
        """保存词表到文件

        Args:
            path: 保存路径
            compress: 是否压缩
        """
        state = {
            "version": 1,
            "token2id": self.token2id,
            "df": self.df,
            "N": self.N,
            "sum_dl": self.sum_dl,
            "idf_arr": self.idf_arr,
        }
        # 将词表状态序列化为二进制数据
        # 使用 HIGHEST_PROTOCOL 确保使用最新且最高效的序列化协议
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

        # 判断传入的path是相对路径还是绝对路径
        # 不包含'/'表示是文件名（相对路径），则使用默认词表目录
        if '/' not in path:
            # 先写入临时文件，写入成功后再重命名，确保原子性操作
            # 避免写入过程中出错导致原文件损坏
            tmp = str(default_vocab_dir) + path + ".tmp"
            path = str(default_vocab_dir) + path
        else:
            # 绝对路径，直接在原路径基础上添加.tmp后缀作为临时文件
            tmp = path + ".tmp"

        # 根据compress参数选择打开方式：压缩使用gzip，否则直接写入
        # 原子性写入：先写入临时文件，成功后通过os.replace原子性地替换目标文件
        with (gzip.open(tmp, "wb") if compress else open(tmp, "wb")) as f:
            f.write(data)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path_or_name: str) -> Optional['Vocabulary']:
        """加载词表

        Args:
            path_or_name: 词表路径或名称

        Returns:
            Vocabulary: 词表对象，加载失败返回None
        """
        try:
            if '/' not in path_or_name:
                path = str(default_vocab_dir) + "/" + path_or_name
            else:
                path = path_or_name

            if not Path(path).exists():
                return None

            with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
                state = pickle.load(f)

            v = cls()
            v.token2id = state["token2id"]
            v.df = state["df"]
            v.N = state["N"]
            v.sum_dl = state.get("sum_dl", 0)
            v.idf_arr = state.get("idf_arr", None)
            return v
        except Exception:
            return None
