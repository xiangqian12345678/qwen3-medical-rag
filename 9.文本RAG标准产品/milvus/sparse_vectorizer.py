"""稀疏向量处理器（BM25）"""
import logging
from typing import List, Dict
from multiprocessing import Pool, cpu_count

import pkuseg

from .embed_config import SparseFieldConfig
from .embed_vocab import Vocabulary

logger = logging.getLogger(__name__)


# 全局变量，用于子进程
_SEG = None


def _init_seg_worker(domain_model: str):
    """子进程启动时加载各自的 pkuseg 实例"""
    global _SEG
    _SEG = pkuseg.pkuseg(model_name=domain_model)


def _cut_worker(text: str) -> List[str]:
    """子进程真正执行的分词函数"""
    return [t.strip() for t in _SEG.cut(text) if t.strip()]


class SparseVectorProcessor:
    """稀疏向量处理器"""

    def __init__(self, vocab: Vocabulary, config: SparseFieldConfig):
        """初始化稀疏向量处理器

        Args:
            vocab: 词表对象
            config: 稀疏向量字段配置
        """
        self.vocab = vocab
        self.config = config
        self.seg = pkuseg.pkuseg(model_name=config.domain_model)
        self.k1 = config.k1
        self.b = config.b

    def tokenize(self, text: str) -> List[str]:
        """单进程分词"""
        return [t.strip() for t in self.seg.cut(text) if t.strip()]

    def tokenize_parallel(self, texts: List[str], chunksize: int = 64):
        """并行分词

        使用多进程池对文本列表进行并行分词，显著提升大批量文本处理速度。
        每个子进程加载独立的 pkuseg 模型实例，通过进程间通信传递分词结果。

        Args:
            texts: 待分词的文本列表
            chunksize: 每个批次的大小，控制任务分配粒度（越大通信开销越小但负载均衡变差）

        Yields:
            List[str]: 逐个产出的分词结果（token列表）
        """
        # 计算实际工作进程数：取配置的workers数和（CPU核心数-1）的较小值
        # 减1是保留一个核心给主进程和其他系统任务，避免CPU占用过高导致系统卡顿
        workers = min(self.config.workers, max(1, cpu_count() - 1))

        # 创建进程池进行并行处理
        # - processes: 工作进程数
        # - initializer: 每个子进程启动时执行的初始化函数，用于加载各自独立的分词器实例
        # - initargs: 传递给初始化函数的参数，这里是领域模型路径
        with Pool(processes=workers, initializer=_init_seg_worker,
                  initargs=(self.config.domain_model,)) as pool:
            # 使用 imap 方法将分词任务分发到各个子进程
            # imap 是惰性求值的迭代器，按需生成结果，节省内存
            # _cut_worker: 子进程中实际执行的分词函数（使用全局变量 _SEG）
            # texts: 输入文本列表
            # chunksize: 每次批量发送给子进程的文本数量，控制任务粒度
            for tokens in pool.imap(_cut_worker, texts, chunksize=chunksize):
                # 逐个产出分词结果（token列表），避免一次性返回所有结果占用大量内存
                yield tokens

    def build_sparse_vector(self, text: str, avgdl: float,
                           update_vocab: bool = False) -> Dict[int, float]:
        """从原始文本构建 BM25 稀疏向量

        Args:
            text: 输入文本
            avgdl: 平均文档长度
            update_vocab: 是否更新词表

        Returns:
            Dict[int, float]: 稀疏向量，key为token_id，value为BM25分数
        """
        tokens = self.tokenize(text)
        return self._build_from_tokens(tokens, avgdl, update_vocab)

    def _build_from_tokens(self, tokens: List[str], avgdl: float,
                           update_vocab: bool = False) -> Dict[int, float]:
        """从 tokens 构建 BM25 稀疏向量

        Args:
            tokens: 分词后的token列表
            avgdl: 平均文档长度
            update_vocab: 是否更新词表

        Returns:
            Dict[int, float]: 稀疏向量
        """
        if update_vocab:
            self.vocab.add_document(tokens)

        tf: Dict[int, int] = {}
        for t in tokens:
            tid = self.vocab.token2id.get(t)
            if tid is None:
                continue
            tf[tid] = tf.get(tid, 0) + 1

        if not tf:
            return {0: 0.0}

        # 计算当前文档长度（文档中各token的词频之和）
        # dl: document length，即当前文档的总词数
        dl = sum(tf.values())

        # 计算文档长度归一化因子 K
        # BM25 通过 K 值对词频进行归一化，避免长文档中频繁出现但意义不大的词获得过高权重
        # 公式: K = k1 * ((1 - b) + b * dl / avgdl)
        #   - k1: 饱和参数，控制词频的上限效应（通常取1.2-2.0）
        #   - b: 长度归一化参数，控制文档长度对分数的影响（0=不归一化，1=完全归一化，通常取0.75）
        #   - dl: 当前文档长度
        #   - avgdl: 语料库平均文档长度
        #   - max(avgdl, 1.0): 防止除零错误，确保分母至少为1
        K = self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1.0))

        # 初始化稀疏向量字典
        # key: token_id（词的整数ID）
        # value: BM25相关性分数
        vec: Dict[int, float] = {}

        # 遍历当前文档中的每个token，计算其BM25分数
        for tid, f in tf.items():
            # 获取当前token的逆文档频率（IDF）
            # IDF 衡量词的区分度：稀有词IDF高，常见词IDF低
            # 通常使用 log((N - df + 0.5) / (df + 0.5)) 计算公式
            #   - N: 文档总数
            #   - df: 包含该词的文档数
            idf = self.vocab.idf(tid)

            # 计算BM25分数
            # 公式: score = IDF * (f * (k1 + 1)) / (f + K)
            #   - f: 当前词在文档中的词频
            #   - (f * (k1 + 1)) / (f + K): 标准化的词频项，考虑了词频饱和和文档长度归一化
            #     当 f 很大时，分数趋于 (k1 + 1)，避免高频词无限制增加分数
            #     当 f 很小时，分数约为 f * (k1 + 1) / K，接近原始词频
            score = idf * (f * (self.k1 + 1.0)) / (f + K)

            # 只保留分数为正的token，过滤掉负分或零分的项
            # 这可以提高检索效率并减少噪声
            if score > 0:
                vec[tid] = float(score)

        return vec if vec else {0: 0.0}

    def vectorize_texts(self, texts: List[str], avgdl: float) -> List[Dict[int, float]]:
        """批量向量化文本

        Args:
            texts: 文本列表
            avgdl: 平均文档长度

        Returns:
            List[Dict[int, float]]: 稀疏向量列表
        """
        return [self.build_sparse_vector(text, avgdl) for text in texts]
