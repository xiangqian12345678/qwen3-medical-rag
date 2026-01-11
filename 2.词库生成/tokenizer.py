"""分词模块 - 基于pkuseg的中文分词与停用词过滤"""

import pkuseg
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from typing import Callable, Iterator, List

# 常用中文停用词
COMMON_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "么", "什么",
    "与", "及", "等", "或", "以", "为", "而", "但", "对", "让", "把", "被", "给", "从", "向", "往",
    "内", "外", "前", "后", "上", "下", "左", "右", "中", "间", "里", "面", "边", "侧", "头", "尾",
    "啊", "吧", "呢", "呀", "哦", "哈", "嘛", "么", "了", "吗", "哎", "唉", "哟", "喂", "嗯", "诶",
    "之", "乎", "者", "也", "矣", "焉", "哉", "兮", "于", "而", "乎", "其", "且", "若", "所", "为",
    "我们", "你们", "他们", "她们", "它们", "咱们", "大家", "各位", "诸位", "本人", "笔者", "作者",
    "因为", "所以", "由于", "因此", "因而", "故", "于是", "不过", "只是", "但是", "然而", "可是",
    "虽然", "尽管", "固然", "即使", "纵使", "哪怕", "无论", "不管", "只要", "除非", "否则", "不然",
    "如果", "假如", "若是", "倘若", "要是", "万一", "以便", "以免", "以免", "使得", "导致", "造成",
}


class Tokenizer:
    """分词器 - 支持单进程和并行分词"""

    def __init__(self, domain_model: str = "medicine"):
        self.domain_model = domain_model
        self.seg = pkuseg.pkuseg(model_name=domain_model)

    def tokenize(self, text: str) -> List[str]:
        """对单条文本进行分词并过滤停用词"""
        tokens = self.seg.cut(text)
        return [t.strip() for t in tokens if t.strip() and t.strip() not in COMMON_STOPWORDS]

    def tokenize_parallel(
        self,
        texts: list,
        workers: int = None,
        chunksize: int = 128
    ) -> Iterator[List[str]]:
        """并行分词 - 流式返回结果"""
        if workers is None:
            workers = max(1, cpu_count() - 1)

        with Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(self.domain_model,)
        ) as pool:
            for tokens in pool.imap(_tokenize_worker, texts, chunksize=chunksize):
                yield tokens


# ====== 多进程 worker ======
_SEG = None


def _init_worker(domain_model: str) -> None:
    """初始化子进程的分词器"""
    global _SEG
    _SEG = pkuseg.pkuseg(model_name=domain_model)


def _tokenize_worker(text: str) -> List[str]:
    """子进程的分词函数"""
    tokens = _SEG.cut(text)
    return [t.strip() for t in tokens if t.strip() and t.strip() not in COMMON_STOPWORDS]
