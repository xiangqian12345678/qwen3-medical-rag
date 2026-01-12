import gzip
import math
import os
import pickle
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Iterable, Iterator

import pkuseg

current_dir = Path(__file__).resolve().parent
default_vocab_dir = str(current_dir) + "/vocab/"

# 确保vocab目录存在
os.makedirs(default_vocab_dir, exist_ok=True)


# ====== 停用词处理 ======
@lru_cache()
def get_stopwords(source="all"):
    """
    获取停用词表
    Args:
        source: 停用词来源，支持:
            - baidu: 百度停用词表
            - hit: 哈工大停用词表
            - ict: 中科院计算所停用词表
            - scu: 四川大学机器智能实验室停用词库
            - cn: 广为流传未知来源的中文停用词表
            - marimo: Marimo multi-lingual stopwords collection 内的中文停用词
            - iso: Stopwords ISO 内的中文停用词
            - all: 上述所有停用词并集
    Returns:
        停用词集合
    """
    supported_source = ["cn", "baidu", "hit", "scu", "marimo", "ict", "iso", "all"]
    if source not in supported_source:
        raise NotImplementedError(f"未知来源: {source}")

    # 获取 stopwords 目录路径
    stopwords_dir = Path(__file__).resolve().parent.parent / "core" / "stopwords"
    stopwords_file = stopwords_dir / f"stopwords.zh.{source}.txt"

    if not stopwords_file.exists():
        raise FileNotFoundError(f"停用词文件不存在: {stopwords_file}")

    return set(stopwords_file.read_text(encoding='utf8').strip().split())


def filter_stopwords(tokens: List[str], source: str = "all") -> List[str]:
    """
    过滤停用词
    Args:
        tokens: 分词结果列表
        source: 停用词来源
    Returns:
        过滤后的词列表
    """
    stopwords_set = get_stopwords(source)
    return [t for t in tokens if t not in stopwords_set]


# ====== worker 全局 ======
_SEG = None  # 每个子进程里各自持有一个分词器


def _init_seg_worker(domain_model: str):
    """
    每个子进程启动时运行：加载各自的 pkuseg 实例
    """
    global _SEG
    import pkuseg as _pk  # 避免主进程/子进程导入冲突
    _SEG = _pk.pkuseg(model_name=domain_model)


def _cut_worker(text: str) -> List[str]:
    """
    子进程真正执行的分词函数
    """
    toks = filter_stopwords(_SEG.cut(text))
    return [t.strip() for t in toks if t.strip()]


class Vocabulary:
    """维护 token->id 与 id->df，用于稀疏向量化"""

    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.df: Dict[int, int] = {}
        self.N: int = 0  # 文档总数
        self.sum_dl: int = 0  # 所有文档长度之和（可选，用于 avgdl）
        # 可选：冻结后缓存
        self.idf_arr: List[float] | None = None

    def add_document(self, tokens: List[str]):
        """
        添加一个文档到词表中，用于构建 TF-IDF/BM25 的词频统计

        Args:
            tokens: 文档经过分词和停用词过滤后的 token 列表

        作用:
            - 更新文档总数 N
            - 更新所有文档长度之和 sum_dl（用于计算平均文档长度 avgdl）
            - 更新 token 到 id 的映射
            - 更新每个 token 的文档频率 df（document frequency）
            - 使缓存的 idf 数组失效（因为词表已改变）
        """
        self.N += 1  # 文档总数递增
        self.sum_dl += len(tokens)  # 累加当前文档长度到总长度

        seen = set()  # 记录本文档中已统计过的 token id，避免同一文档内重复计数

        for t in tokens:  # 遍历文档中的所有 token
            # 如果 token 不在词表中，分配新的 token id
            if t not in self.token2id:
                self.token2id[t] = len(self.token2id)

            tid = self.token2id[t]  # 获取 token id

            # 只有在本文档中第一次出现的 token 才计入文档频率 df
            # 这样确保 df 计算的是"包含该 token 的文档数"，而不是"该 token 在所有文档中的总出现次数"
            if tid not in seen:
                self.df[tid] = self.df.get(tid, 0) + 1
                seen.add(tid)

        # 词表改变后，旧的 idf 缓存作废，下次需要重新计算
        self.idf_arr = None

    def idf(self, tid: int) -> float:
        """
        计算 token 的逆文档频率 (IDF, Inverse Document Frequency)

        Args:
            tid: token 的 id

        Returns:
            IDF 值，表示该 token 在整个语料库中的稀有程度

        公式说明:
            使用 BM25 的 IDF 公式：
            IDF = log(1 + (N - df + 0.5) / (df + 0.5))

            其中:
            - N: 文档总数
            - df: 包含该 token 的文档数
            - 0.5: 平滑因子，避免 df=0 或 df=N 时出现 0 或无穷大

        特点:
            - df 越小（出现该 token 的文档越少），IDF 越大，token 越有区分度
            - df 越大（出现该 token 的文档越多），IDF 越小，token 越普遍
        """
        # 如果已经计算并缓存了所有 token 的 IDF 数组，直接返回
        # 这是在 freeze() 方法被调用后，预计算所有 token 的 IDF 值以加速查询
        if self.idf_arr is not None:
            return self.idf_arr[tid]

        # 动态计算：获取该 token 的文档频率
        df = self.df.get(tid, 0)

        # 使用 BM25 的 IDF 公式计算
        # (N - df + 0.5) / (df + 0.5): 不包含该 token 的文档数 / 包含该 token 的文档数（平滑后）
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def freeze(self):
        """建库结束后一次性缓存 idf，加速查询"""
        V = len(self.token2id)
        df_arr = [0] * V
        for tid, c in self.df.items():
            df_arr[tid] = c
        self.idf_arr = [math.log(1.0 + (self.N - d + 0.5) / (d + 0.5)) for d in df_arr]

    def save(self, path: str, compress: bool = True):
        """
        保存词表到文件

        Args:
            path: 保存路径
                - 如果不包含 '/'，则作为文件名保存到默认的 vocab 目录
                - 如果包含 '/'，则作为完整路径（绝对或相对路径）
            compress: 是否使用 gzip 压缩，默认为 True

        特点:
            - 使用 pickle 序列化词表数据
            - 先写入临时文件，完成后再原子替换，避免写入失败损坏原文件
            - 支持压缩以减少磁盘占用
        """
        # 构建要保存的状态字典
        state = {
            "version": 1,  # 版本号，用于未来兼容性检查
            "token2id": self.token2id,  # token 到 id 的映射
            "df": self.df,  # token id 到文档频率的映射
            "N": self.N,  # 文档总数
            "sum_dl": self.sum_dl,  # 所有文档长度之和
            # 可选缓存，存在就一起存
            "idf_arr": self.idf_arr,  # 预计算的 IDF 数组（如果已冻结）
        }

        # 使用 pickle 序列化状态字典
        # HIGHEST_PROTOCOL 使用最新协议，提供更好的性能和压缩
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

        # 处理路径：区分文件名和完整路径
        if '/' not in path:  # 没有自定义绝对路径，自动保存在当前目录下的 vocab 文件夹
            # 构建临时文件路径（先写入 .tmp 文件）
            tmp = str(default_vocab_dir) + path + ".tmp"
            # 构建最终文件路径
            path = str(default_vocab_dir) + path
        else:  # 用户提供了完整路径
            # 在原路径后添加 .tmp 后缀作为临时文件
            tmp = path + ".tmp"

        # 写入文件：根据 compress 参数选择是否使用 gzip 压缩
        with (gzip.open(tmp, "wb") if compress else open(tmp, "wb")) as f:
            f.write(data)

        # 原子替换：使用 os.replace 原子性地将临时文件重命名为目标文件
        # 这样可以防止在写入过程中程序崩溃导致原文件损坏
        os.replace(tmp, path)

    @classmethod
    def load(cls, path_or_name: str):
        try:
            if '/' not in path_or_name:  # 直接传入的名字
                path = str(default_vocab_dir) + "/" + path_or_name
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
        except Exception as e:
            return None


# ====== 并行分词 + BM25 ======
class BM25Vectorizer:
    def __init__(self, vocab: Vocabulary, domain_model: str = "medicine", k1: float = 1.5, b: float = 0.75):
        # 单进程下仍可直接用
        self.seg = pkuseg.pkuseg(model_name=domain_model)
        self.domain_model = domain_model
        self.vocab = vocab
        # BM25 参数
        self.k1 = k1
        self.b = b

    # --- 单进程分词 ---
    def tokenize(self, text: str) -> List[str]:
        return [t.strip() for t in filter_stopwords(self.seg.cut(text)) if t.strip()]

    # --- 多进程分词（批处理/流式产出）---
    def tokenize_parallel(
            self,
            texts: Iterable[str],
            workers: int = None,
            chunksize: int = 64
    ) -> Iterator[List[str]]:
        """
        并行分词：按 chunksize 批量发给子进程，流式返回 tokens 列表。

        Args:
            texts: 可迭代的文本集合（如列表、生成器等）
            workers: 子进程数量，默认为 CPU 核心数-1
            chunksize: 每次分发给子进程的批次大小，默认为 64

        Returns:
            Iterator[List[str]]: 迭代器，逐批产出分词后的 token 列表

        特点:
            - 使用进程池实现并行分词，充分利用多核 CPU
            - pool.imap 是惰性求值，内存占用稳定，适合处理大规模数据
            - 子进程独立持有 pkuseg 分词器实例，避免进程间通信开销
            - 流式返回结果，避免一次性加载所有数据到内存
        """
        if workers is None:
            # 默认使用 CPU 核心数减 1，留一个核心给主进程
            workers = max(1, cpu_count() - 1)

        with Pool(
                processes=workers,
                initializer=_init_seg_worker,  # 每个子进程启动时加载分词器
                initargs=(self.domain_model,)  # 传入领域模型参数
        ) as pool:
            # imap 是流式的，内存占用更稳
            # 按批次处理，每批 chunksize 个文本发给子进程
            for tokens in pool.imap(_cut_worker, texts, chunksize=chunksize):
                yield tokens  # 流式产出结果

    def build_sparse_vec_from_tokens(
            self,
            tokens: List[str],
            avgdl: float,
            update_vocab: bool = False
    ) -> Dict[int, float]:
        """
        允许传入已分好的 tokens（建议并行切好后再喂这里）

        功能：根据传入的 tokens 列表，构建 BM25 稀疏向量（token_id -> score 的字典）

        参数说明：
            tokens: List[str]      - 已分词的 token 列表
            avgdl: float           - 平均文档长度（用于 BM25 归一化）
            update_vocab: bool     - 是否更新词表（建库阶段为 True，查询阶段为 False）

        应用场景：
            建库阶段（update_vocab=True）：构建索引时更新词表统计，生成稀疏向量存入向量库
            查询阶段（update_vocab=False）：仅对查询文本进行向量化，不修改词表
        """
        # 1. 更新词表（可选，仅建库阶段需要）
        if update_vocab:
            self.vocab.add_document(tokens)

        # 2. 词频统计（容忍 OOV - Out of Vocabulary）
        tf: Dict[int, int] = {}
        for t in tokens:
            tid = self.vocab.token2id.get(t)  # token 不存在时返回 None，跳过
            if tid is None:
                continue
            tf[tid] = tf.get(tid, 0) + 1  # 统计 token 在当前文档中的出现次数

        # 3. 空文档处理
        if not tf:
            return {}

        # 4. 计算文档长度
        dl = sum(tf.values())

        # 5. 计算 BM25 归一化系数 K
        #    公式：K = k1 * (1 - b + b * dl / avgdl)
        #    其中：
        #      - k1: 词频饱和参数（典型值 1.2-2.0），控制词频对分数的线性增长上限
        #      - b: 长度归一化参数（典型值 0.75），控制文档长度对分数的影响程度
        #      - dl: 当前文档长度
        #      - avgdl: 语料库平均文档长度
        K = self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1.0))

        # 6. 对每个 token 计算 BM25 分数
        vec: Dict[int, float] = {}
        for tid, f in tf.items():
            idf = self.vocab.idf(tid)  # 获取逆文档频率（衡量 token 的稀有程度）
            score = idf * (f * (self.k1 + 1.0)) / (f + K)  # BM25 公式
            if score > 0:
                vec[tid] = float(score)
        return vec if vec is not None else {"0": 0.0}

    def build_sparse_vec(self, text: str, avgdl: float, update_vocab: bool = False):
        """
        从原始文本构建 BM25 稀疏向量（内部调用分词+向量化）

        参数说明：
            text: str               - 原始文本字符串
            avgdl: float            - 平均文档长度（用于 BM25 归一化）
            update_vocab: bool      - 是否更新词表（建库阶段为 True，查询阶段为 False）

        处理流程：
            1. 对文本进行分词和停用词过滤
            2. 调用 build_sparse_vec_from_tokens 计算 BM25 稀疏向量

        与 build_sparse_vec_from_tokens 的区别：
            - build_sparse_vec: 接收原始文本，内部自动分词（适合单次调用）
            - build_sparse_vec_from_tokens: 接收已分词的 tokens（适合批量并行处理，避免重复分词）
        """
        # 1. 对文本进行分词和停用词过滤
        tokens = self.tokenize(text)

        # 2. 调用 build_sparse_vec_from_tokens 方法，从 tokens 构建 BM25 稀疏向量
        return self.build_sparse_vec_from_tokens(tokens, avgdl, update_vocab)

    def vectorize_texts(self, texts: List[str], avgdl) -> List[Dict[int, float]]:
        vecs = []
        for i in range(len(texts)):
            tokens = self.tokenize(texts[i])
            vecs.append(self.build_sparse_vec_from_tokens(tokens, avgdl))
        return vecs
