"""数据导入工具 - 将文档数据导入知识库"""
import logging
from typing import List, Optional, Callable
from pathlib import Path

from langchain_core.documents import Document
from tqdm import tqdm

from knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class DataIngestor:
    """数据导入器"""

    def __init__(self, knowledge_base: KnowledgeBase):
        """初始化数据导入器

        Args:
            knowledge_base: 知识库实例
        """
        self.kb = knowledge_base

    def load_from_documents(
            self,
            documents: List[Document],
            show_progress: bool = True,
            batch_size: int = 100
    ) -> int:
        """从文档列表加载数据

        Args:
            documents: 文档列表
            show_progress: 是否显示进度条
            batch_size: 批次大小

        Returns:
            int: 插入的总行数
        """
        total_rows = 0

        if show_progress:
            pbar = tqdm(total=len(documents), desc="导入文档", unit="doc")
        else:
            pbar = None

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            rows = self.kb.add_documents(batch, show_progress=False)
            total_rows += rows

            if pbar:
                pbar.update(len(batch))

        if pbar:
            pbar.close()

        logger.info(f"导入完成: {len(documents)} 个文档, {total_rows} 行")
        return total_rows

    def load_from_file(
            self,
            file_path: str,
            loader: Callable,
            show_progress: bool = True,
            **kwargs
    ) -> int:
        """从文件加载数据

        Args:
            file_path: 文件路径
            loader: 文档加载函数
            show_progress: 是否显示进度条
            **kwargs: 传递给 loader 的参数

        Returns:
            int: 插入的总行数
        """
        documents = loader(file_path, **kwargs)
        return self.load_from_documents(documents, show_progress=show_progress)

    def load_from_directory(
            self,
            directory: str,
            loader: Callable,
            show_progress: bool = True,
            file_pattern: str = "*",
            **kwargs
    ) -> int:
        """从目录加载数据

        Args:
            directory: 目录路径
            loader: 文档加载函数
            show_progress: 是否显示进度条
            file_pattern: 文件匹配模式
            **kwargs: 传递给 loader 的参数

        Returns:
            int: 插入的总行数
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        all_documents = []
        files = list(dir_path.glob(file_pattern))

        for file_path in tqdm(files, desc="加载文件", disable=not show_progress):
            documents = loader(str(file_path), **kwargs)
            all_documents.extend(documents)

        return self.load_from_documents(all_documents, show_progress=show_progress)


def create_sample_documents(count: int = 10) -> List[Document]:
    """
    创建示例文档用于测试

    Args:
        count: 文档数量

    Returns:
        List[Document]: 示例文档列表
    """
    import hashlib

    documents = []

    for i in range(count):
        chunk_id = i
        chunk = f"这是第 {i} 个文档块的内容。医学知识：高血压是一种常见的慢性疾病，需要长期管理。"

        parent_chunk = f"父级文档块 {i // 2}：包含了多个相关联的子文档块，用于提供更全面的上下文信息。"

        summary = f"摘要：本文档块主要介绍了医学知识 {i}"

        questions = [
            f"什么是高血压{i}？",
            f"如何管理高血压{i}？"
        ]

        document = f"完整文档 {i // 3}"

        source = "示例数据"
        source_name = "sample"
        lt_doc_id = f"doc_{i // 3}"
        hash_id = hashlib.md5(chunk.encode()).hexdigest()

        metadata = {
            "chunk": chunk,
            "parent_chunk": parent_chunk,
            "summary": summary,
            "questions": questions,
            "document": document,
            "source": source,
            "source_name": source_name,
            "lt_doc_id": lt_doc_id,
            "chunk_id": chunk_id,
            "hash_id": hash_id
        }

        doc = Document(page_content=chunk, metadata=metadata)
        documents.append(doc)

    return documents


# =============================================================================
# 快速入口函数
# =============================================================================

def quick_ingest(
        config_path: Optional[str] = None,
        documents: Optional[List[Document]] = None,
        init_kb: bool = True
) -> KnowledgeBase:
    """快速导入数据到知识库

    Args:
        config_path: 配置文件路径
        documents: 文档列表，如果为 None 则使用示例文档
        init_kb: 是否初始化知识库

    Returns:
        KnowledgeBase: 知识库实例
    """
    # 创建知识库
    kb = KnowledgeBase(config_path)

    # 初始化知识库
    if init_kb:
        kb.initialize()

    # 加载文档
    if documents is None:
        documents = create_sample_documents(10)

    ingestor = DataIngestor(kb)
    ingestor.load_from_documents(documents)

    return kb
