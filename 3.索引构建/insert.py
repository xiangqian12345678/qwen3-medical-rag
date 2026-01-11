"""数据插入工具"""
from __future__ import annotations

from typing import List, Dict, Any, Iterable, Optional, Callable

from pymilvus import MilvusClient
from tqdm import tqdm


def _chunks(seq: Iterable[Any], size: int):
    """将序列分割成固定大小的批次生成器

    Args:
        seq: 可迭代对象，待分割的序列
        size: 每批的大小

    Yields:
        List[Any]: 包含最多 size 个元素的列表
    """
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= max(1, size):
            yield buf
            buf = []
    if buf:
        yield buf


def _maybe_update_progress(pbar: Optional[tqdm], progress_fn: Optional[Callable],
                           inc: int, done_total: List[int]):
    """统一进度更新

    Args:
        pbar: tqdm 实例或 None
        progress_fn: 回调或 None
        inc: 本次新增完成数量
        done_total: 长度为2的列表 [done, total]，以可变引用传入
    """
    done_total[0] += inc
    if pbar is not None:
        pbar.update(inc)
    if progress_fn:
        try:
            progress_fn(done_total[0], done_total[1])
        except Exception:
            pass


def insert_rows(
        client: MilvusClient,
        collection_name: str,
        rows: List[Dict[str, Any]],
        show_progress: bool = False,
        progress_fn: Optional[Callable[[int, int], None]] = None,
):
    """批量插入数据到 Milvus 集合

    Args:
        client: Milvus 客户端实例
        collection_name: 目标集合名称
        rows: 待插入的数据行列表
        show_progress: 是否显示进度条
        progress_fn: 进度回调函数
    rows样例：
        rows = [
            {
                # 主键字段（必填）
                "pk": "doc_hash_001_chunk_0",

                # 基础字段
                "chunk": "糖尿病是一种慢性代谢性疾病，主要特征是高血糖。长期高血糖可导致各种并发症。",
                "parent_chunk": "糖尿病是一种慢性代谢性疾病，主要特征是高血糖。长期高血糖可导致各种并发症，包括心血管疾病、肾病、视网膜病变等。",
                "summary": "糖尿病是高血糖特征的慢性病，可引发多种并发症。",
                "questions": ["什么是糖尿病?", "糖尿病的主要特征是什么?"],
                "document": "糖尿病是一种慢性代谢性疾病...",
                "source": "/data/medical/diabetes.pdf",
                "source_name": "糖尿病知识手册.pdf",
                "lt_doc_id": "doc_001",
                "chunk_id": 0,
                "hash_id": "doc_hash_001",

                # 稠密向量字段（1024维，COSINE距离）
                "chunk_dense": [0.0123, 0.0234, 0.0345, ...],  # 1024维向量
                "parent_chunk_dense": [0.0456, 0.0567, 0.0678, ...],
                # summary_dense 未启用向量化，不需要提供

                # questions 字段是 list 类型，对应每个问题一个向量
                "questions_dense": [[0.0789, 0.0890, 0.0901, ...], [0.0912, 0.0923, 0.0934, ...]],

                # 稀疏向量字段（BM25算法）
                "chunk_sparse": {"糖尿病": 1.5, "慢性": 0.8, "疾病": 0.6, ...}
            },
            {
                "pk": "doc_hash_002_chunk_0",
                "chunk": "高血压需要长期药物治疗，常用降压药包括钙通道阻滞剂、ACE抑制剂等。",
                "parent_chunk": "高血压需要长期药物治疗。常用降压药包括钙通道阻滞剂、ACE抑制剂、利尿剂等。",
                "summary": "高血压需长期用药，常用钙通道阻滞剂等药物。",
                "questions": ["高血压如何治疗?", "常用的降压药有哪些?"],
                "document": "高血压治疗指南...",
                "source": "/data/medical/hypertension.pdf",
                "source_name": "高血压治疗指南.pdf",
                "lt_doc_id": "doc_002",
                "chunk_id": 0,
                "hash_id": "doc_hash_002",
                "chunk_dense": [0.0234, 0.0345, 0.0456, ...],
                "parent_chunk_dense": [0.0567, 0.0678, 0.0789, ...],
                "questions_dense": [[0.0901, 0.0912, 0.0923, ...], [0.0934, 0.0945, 0.0956, ...]],
                "chunk_sparse": {"高血压": 1.8, "降压": 1.2, "药物": 0.9, ...}
            }
        ]

    """
    bs = 20
    total = len(rows)
    done_total = [0, total]

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total, desc="Milvus insert", unit="row")

    try:
        for batch in _chunks(rows, bs):
            _ = client.insert(collection_name=collection_name, data=batch)
            _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
    finally:
        if pbar is not None:
            pbar.close()


def upsert_rows(
        client: MilvusClient,
        collection_name: str,
        rows: List[Dict[str, Any]],
        show_progress: bool = False,
        progress_fn: Optional[Callable[[int, int], None]] = None,
):
    """批量插入或更新数据到 Milvus 集合

    Args:
        client: Milvus 客户端实例
        collection_name: 目标集合名称
        rows: 待插入的数据行列表，必须包含 "pk" 字段
        show_progress: 是否显示进度条
        progress_fn: 进度回调函数
    """
    bs = 20
    total = len(rows)
    done_total = [0, total]

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total, desc="Milvus upsert", unit="row")

    try:
        if getattr(client, "upsert", None):
            for batch in _chunks(rows, bs):
                _ = client.upsert(collection_name=collection_name, data=batch)
                _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
        else:
            for batch in _chunks(rows, bs):
                _ = client.insert(collection_name=collection_name, data=batch)
                _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
    finally:
        if pbar is not None:
            pbar.close()


def delete_by_ids(
        client: MilvusClient,
        collection_name: str,
        ids: List[str],
        id_field: str = "pk",
        *,
        show_progress: bool = False,
        progress_fn: Optional[Callable[[int, int], None]] = None,
):
    """批量删除 Milvus 集合中的数据

    Args:
        client: Milvus 客户端实例
        collection_name: 目标集合名称
        ids: 待删除的 ID 列表
        id_field: ID 字段名
        show_progress: 是否显示进度条
        progress_fn: 进度回调函数
    """
    bs = 100
    total = len(ids)
    done_total = [0, total]

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total, desc="Milvus delete", unit="id")

    try:
        for batch in _chunks(ids, bs):
            id_list = ", ".join(f'"{i}"' for i in batch)
            expr = f'{id_field} in [{id_list}]'
            client.delete(collection_name=collection_name, expr=expr)
            _maybe_update_progress(pbar, progress_fn, len(batch), done_total)
    finally:
        if pbar is not None:
            pbar.close()
