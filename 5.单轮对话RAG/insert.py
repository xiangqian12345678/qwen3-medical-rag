"""数据插入工具"""
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pymilvus import MilvusClient


logger = logging.getLogger(__name__)


def insert_rows(
    client: MilvusClient,
    collection_name: str,
    rows: List[Dict[str, Any]],
    show_progress: bool = False
) -> int:
    """
    批量插入数据到Milvus

    Args:
        client: Milvus客户端
        collection_name: 集合名称
        rows: 数据行列表
        show_progress: 是否显示进度条

    Returns:
        插入的行数
    """
    if not rows:
        return 0

    batch_size = 100
    total = 0

    iterator = range(0, len(rows), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"插入数据到 {collection_name}")

    for i in iterator:
        batch = rows[i:i + batch_size]
        client.insert(collection_name=collection_name, data=batch)
        total += len(batch)

    logger.info(f"成功插入 {total} 行数据到 {collection_name}")
    return total
