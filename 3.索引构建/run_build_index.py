"""医疗知识库索引构建脚本
从 output/annotation 目录读取数据并构建 Milvus 向量索引
"""
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from langchain_core.documents import Document

from knowledge_base import KnowledgeBase
from ingest import DataIngestor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据加载器
# =============================================================================

class AnnotationDataLoader:
    """标注数据加载器"""

    def __init__(self, data_dir: str = "../output/annotation"):
        """初始化数据加载器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    def load_from_jsonl(self, file_path: str) -> List[Document]:
        """从 JSONL 文件加载数据

        Args:
            file_path: JSONL 文件路径

        Returns:
            List[Document]: 文档列表
        """
        documents = []
        file_path = Path(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"加载 {file_path.name}", unit="行"):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                doc = self._parse_annotation_data(data, source_name=file_path.stem)
                if doc:
                    documents.append(doc)

        logger.info(f"从 {file_path.name} 加载了 {len(documents)} 个文档")
        return documents

    def load_from_directory(self, pattern: str = "*.jsonl") -> List[Document]:
        """从目录加载所有 JSONL 文件

        Args:
            pattern: 文件匹配模式

        Returns:
            List[Document]: 文档列表
        """
        all_documents = []
        files = list(self.data_dir.glob(pattern))

        if not files:
            logger.warning(f"目录 {self.data_dir} 中没有找到匹配 {pattern} 的文件")
            return all_documents

        for file_path in files:
            documents = self.load_from_jsonl(str(file_path))
            all_documents.extend(documents)

        return all_documents

    def _parse_annotation_data(self, data: dict, source_name: str) -> Optional[Document]:
        """解析标注数据为 Document

        Args:
            data: 原始标注数据
            source_name: 数据源名称

        Returns:
            Document: 解析后的文档
        """
        try:
            chunk = data.get('chunk', '')
            parent_chunk = data.get('parent_chunk', '')
            summary = data.get('summary', '')
            questions = data.get('questions', [])
            chunk_id = data.get('chunk_id', 0)

            # 跳过空 chunk
            if not chunk or not chunk.strip():
                return None

            # 构建文档名（基于 chunk_id）
            document = f"document_{chunk_id // 100}" if chunk_id else "document_0"

            # 计算哈希 ID
            # 主要作用：
            # 1.构建唯一主键 pk
            # 2.在多向量展开场景中作为 origin_pk 追踪原始文档
            #   origin_pk = doc_dict.get("hash_id", "")
            #   if not origin_pk:
            #       origin_pk = self._compute_hash_id(document)
            # 3.保存为元数据供查询使用
            hash_id = hashlib.md5(chunk.encode('utf-8')).hexdigest()

            # 构建 lt_doc_id（同一文档的 chunks 共享相同的 lt_doc_id）
            lt_doc_id = f"lt_doc_{chunk_id // 10}" if chunk_id else "lt_doc_0"

            # 构建 pk（主键） 唯一标识：每个文档块有唯一的 pk ，避免重复插
            pk = f"{hash_id[:16]}_{chunk_id}"

            metadata = {
                "pk": pk,
                "chunk": chunk,
                "parent_chunk": parent_chunk,
                "summary": summary,
                "questions": questions if questions else [],
                "document": document,
                "source": str(self.data_dir),
                "source_name": source_name,
                "lt_doc_id": lt_doc_id,
                "chunk_id": chunk_id,
                "hash_id": hash_id
            }

            return Document(page_content=chunk, metadata=metadata)

        except Exception as e:
            logger.warning(f"解析数据失败: {e}, 数据: {data}")
            return None


# =============================================================================
# 索引构建器
# =============================================================================

class IndexBuilder:
    """索引构建器"""

    def __init__(self, config_path: str = "index.yaml"):
        """初始化索引构建器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.kb = KnowledgeBase(config_path)
        self.ingestor = DataIngestor(self.kb)

    def build_index(
            self,
            data_dir: str = "../output/annotation",
            file_pattern: str = "*.jsonl",
            init_kb: bool = True,
            batch_size: int = 100
    ):
        """构建索引

        Args:
            data_dir: 数据目录
            file_pattern: 文件匹配模式
            init_kb: 是否初始化知识库（删除旧数据重建）
            batch_size: 批处理大小
        """
        # 初始化知识库
        if init_kb:
            logger.info("初始化知识库...")
            self.kb.initialize()

        # 加载数据
        logger.info(f"从 {data_dir} 加载数据...")
        loader = AnnotationDataLoader(data_dir)
        documents = loader.load_from_directory(file_pattern)

        if not documents:
            logger.warning("没有文档需要索引")
            return

        # 导入数据
        logger.info(f"开始导入 {len(documents)} 个文档...")
        total_rows = self.ingestor.load_from_documents(
            documents,
            show_progress=True,
            batch_size=batch_size
        )

        logger.info(f"索引构建完成！共导入 {len(documents)} 个文档, {total_rows} 行数据")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="医疗知识库索引构建")
    parser.add_argument(
        "--config",
        type=str,
        default="index.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../output/annotation",
        help="数据目录路径"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.jsonl",
        help="文件匹配模式"
    )
    parser.add_argument(
        "--no-init",
        action="store_true",
        help="不初始化知识库（不删除旧数据）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="批处理大小"
    )

    args = parser.parse_args()

    # 构建索引
    builder = IndexBuilder(args.config)
    builder.build_index(
        data_dir=args.data_dir,
        file_pattern=args.file_pattern,
        init_kb=not args.no_init,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
