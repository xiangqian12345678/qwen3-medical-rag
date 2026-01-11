"""文档向量化器（处理多向量字段）"""
import logging
from typing import List, Dict, Any, Optional
from copy import deepcopy
from langchain_core.documents import Document

from config import IndexConfig
from embedding_client import create_embedding_client
from sparse_vectorizer import SparseVectorProcessor
from vocab import Vocabulary

logger = logging.getLogger(__name__)


class VectorFieldProcessor:
    """向量字段处理器基类"""

    def __init__(self, config: IndexConfig):
        """初始化向量字段处理器

        Args:
            config: 索引配置对象，包含稠密和稀疏字段的配置信息
        """
        self.config = config
        # 稠密向量嵌入器字典：key为字段名，value为嵌入器实例
        # 支持同时为多个字段配置不同的稠密向量模型（如 title 用轻量模型，chunk 用重模型）
        self.dense_embedders: Dict[str, Any] = {}
        # 稀疏向量处理器（BM25）：全库共享一个词表和处理器
        # 稀疏向量基于词表计算，通常只需要一个处理器处理所有稀疏字段
        self.sparse_processor: Optional[SparseVectorProcessor] = None

        # 初始化稠密向量嵌入器
        # 遍历配置中的所有稠密字段，为每个标记为 embed=True 的字段创建对应的嵌入器
        for field_name, field_config in config.dense_fields.items():
            if field_config.embed:
                # 根据字段配置（模型类型、API密钥等）创建相应的嵌入客户端
                # 支持多种嵌入服务：OpenAI、通义千问、本地模型等
                self.dense_embedders[field_name] = create_embedding_client(field_config)

        # 初始化稀疏向量处理器（BM25）
        # 注意：稀疏处理器只初始化一次，用于所有稀疏字段
        # 因为 BM25 基于统一的词表统计 IDF，必须共享同一个词表
        for field_name, field_config in config.sparse_fields.items():
            if field_config.embed:
                # 从指定路径加载预构建的词表
                # 词表包含：token2id 映射、文档频率统计、IDF 值、平均文档长度等
                vocab = Vocabulary.load(field_config.vocab_path)
                if vocab is None:
                    # 词表加载失败时抛出异常，避免使用未初始化的词表导致计算错误
                    raise FileNotFoundError(f"词表加载失败: {field_config.vocab_path}")
                # 创建稀疏向量处理器，传入词表和字段配置（包含 k1、b 等 BM25 参数）
                # 初始化后即可对任意文本进行分词并计算 BM25 稀疏向量
                self.sparse_processor = SparseVectorProcessor(vocab, field_config)

    def get_dense_embedding(self, field_name: str, text: str) -> List[float]:
        """获取稠密向量嵌入

        Args:
            field_name: 字段名称
            text: 输入文本

        Returns:
            List[float]: 向量嵌入
        """
        if field_name not in self.dense_embedders:
            raise ValueError(f"字段 {field_name} 未配置稠密向量")
        return self.dense_embedders[field_name].embed_query(text)

    def get_dense_embeddings_batch(self, field_name: str, texts: List[str]) -> List[List[float]]:
        """批量获取稠密向量嵌入

        Args:
            field_name: 字段名称
            texts: 输入文本列表

        Returns:
            List[List[float]]: 向量嵌入列表
        """
        if field_name not in self.dense_embedders:
            raise ValueError(f"字段 {field_name} 未配置稠密向量")
        return self.dense_embedders[field_name].embed_documents(texts)

    def get_sparse_vector(self, text: str, field_name: str = "chunk") -> Dict[int, float]:
        """获取稀疏向量

        Args:
            text: 输入文本
            field_name: 字段名称（用于获取配置）

        Returns:
            Dict[int, float]: 稀疏向量
        """
        if self.sparse_processor is None:
            raise ValueError("未配置稀疏向量处理器")
        if field_name not in self.config.sparse_fields:
            raise ValueError(f"字段 {field_name} 未配置稀疏向量")

        field_config = self.config.sparse_fields[field_name]
        avgdl = self.sparse_processor.vocab.avgdl if self.sparse_processor.vocab.N > 0 else 100.0
        return self.sparse_processor.build_sparse_vector(text, avgdl)


class DocumentVectorizer:
    """文档向量化器 - 处理单行多向量字段展开"""

    def __init__(self, config: IndexConfig):
        """初始化文档向量化器

        Args:
            config: 索引配置
        """
        self.config = config
        self.field_processor = VectorFieldProcessor(config)

    def vectorize_document(self, document: Document) -> List[Dict[str, Any]]:
        """向量化单个文档，处理多向量字段展开

        对于 list 类型的字段（如 questions），每个元素会展开为单独的行。

        Args:
            document: LangChain Document对象

        Returns:
            List[Dict[str, Any]]: 向量化后的行列表
        """
        # 收集所有需要展开的向量字段
        list_fields = []
        str_fields = []

        # 遍历配置中的所有稠密字段（需要向量化的字段）
        for field_name, field_config in self.config.dense_fields.items():
            # 如果该字段不需要嵌入，跳过
            if not field_config.embed:
                continue
            # 将字段按类型分类：list类型字段和字符串类型字段分别处理
            if field_config.type == "list":
                list_fields.append(field_name)
            else:
                str_fields.append(field_name)

        # 深拷贝文档的元数据，避免修改原始文档对象
        # metadata 包含文档的各种属性，如标题、来源、标签等
        doc_dict = deepcopy(document.metadata)
        # 获取文档的主要内容文本
        page_content = document.page_content

        # 将文本内容设置到对应的基础字段（通常是 chunk 字段）
        # chunk 字段用于存储分片后的文本片段
        if "chunk" in self.config.base_fields:
            doc_dict["chunk"] = page_content

        # 获取文档的主键（origin_pk），用于标识原始文档
        # 优先使用已有的 hash_id，如果没有则重新计算
        origin_pk = doc_dict.get("hash_id", "")
        if not origin_pk:
            origin_pk = self._compute_hash_id(document)

        rows = []

        # 处理需要展开的 list 类型字段
        # 对于 list 字段，每个元素都会生成一行记录
        for list_field in list_fields:
            # 获取 list 字段的值，如果不是列表类型则转换为单元素列表
            field_value = doc_dict.get(list_field, [])
            if not isinstance(field_value, list):
                field_value = [field_value] if field_value else []

            # 遍历 list 中的每个元素，为每个元素创建一行记录
            for vector_id, item in enumerate(field_value):
                # 跳过空值元素
                if not item:
                    continue

                # 创建基础行数据，包含文档的基础信息和元数据
                row = self._create_base_row(doc_dict, origin_pk, vector_id)

                # 设置当前 list 字段的文本内容及其对应的稠密向量
                field_config = self.config.dense_fields[list_field]
                index_field = field_config.index_field  # 向量字段名

                row[list_field] = str(item)  # 存储文本内容
                row[index_field] = self.field_processor.get_dense_embedding(list_field, str(item))  # 存储嵌入向量

                # 其他 str 类型字段的向量置为零向量
                # 这是为了保持所有行的向量维度一致，便于存储和计算
                for sf in str_fields:
                    sf_config = self.config.dense_fields[sf]
                    row[sf_config.index_field] = [0.0] * sf_config.dimension

                # 稀疏向量置空（list 字段通常不需要稀疏向量）
                # 稀疏向量一般用于 chunk 字段的 BM25 检索
                sparse_config = self.config.sparse_fields.get("chunk")
                if sparse_config and sparse_config.embed:
                    row[sparse_config.index_field] = {}

                rows.append(row)

        # 处理 str 类型字段（作为主行）
        # str 字段通常包含文档的主要内容，如 chunk、title 等
        if str_fields:
            # 创建主行数据，vector_id 为 -1 表示这是主行
            row = self._create_base_row(doc_dict, origin_pk, -1)

            # 填充所有 str 字段的向量
            for str_field in str_fields:
                field_config = self.config.dense_fields[str_field]
                index_field = field_config.index_field

                # 获取字段的文本值，如果为空则使用默认值
                text_value = doc_dict.get(str_field, "")
                if not text_value:
                    # chunk 字段默认使用文档主要内容
                    text_value = page_content if str_field == "chunk" else ""

                row[str_field] = text_value  # 存储文本内容
                row[index_field] = self.field_processor.get_dense_embedding(str_field, text_value)  # 存储嵌入向量

            # 处理稀疏向量（用于 BM25 检索）
            # 稀疏向量基于词频统计，用于关键词匹配检索
            sparse_config = self.config.sparse_fields.get("chunk")
            if sparse_config and sparse_config.embed:
                # 使用 chunk 字段的内容计算稀疏向量
                text_for_sparse = doc_dict.get("chunk", page_content)
                row[sparse_config.index_field] = self.field_processor.get_sparse_vector(text_for_sparse, "chunk")

            rows.append(row)

        # 如果没有生成任何行（没有需要向量化的字段），返回一个只包含基础信息的行
        return rows if rows else [self._create_base_row(doc_dict, origin_pk, -1)]

    def _create_base_row(self, doc_dict: Dict[str, Any], origin_pk: str, vector_id: int) -> Dict[str, Any]:
        """创建基础行

        该方法用于构建 Milvus 集合中的一条记录的基础结构。对于包含多向量字段的文档
        （如 questions 列表类型字段），需要将其展开为多条记录，每条记录对应一个向量索引。

        Args:
            doc_dict: 文档元数据字典，包含 chunk、parent_chunk、summary、questions、
                      document、source、source_name、lt_doc_id、chunk_id、hash_id 等字段
            origin_pk: 原始主键，通常为文档的 hash_id，用于追踪同一文档的多条展开记录
            vector_id: 向量ID，用于标识多向量字段中的具体元素
                       - vector_id = -1: 主行，存储 str 类型字段的向量和稀疏向量
                       - vector_id >= 0: 展开行，存储 list 类型字段中第 vector_id 个元素的向量
        Returns:
            Dict[str, Any]: 基础行，包含以下内容：
                - pk: 主键（若 auto_id=False），格式为 "{origin_pk}" 或 "{origin_pk}_v{vector_id}"
                - origin_pk: 原始主键，用于追溯同一文档的多条展开记录
                - vector_id: 向量ID，用于标识该行对应的向量元素
                - 基础字段: chunk、parent_chunk、summary、questions 等字段的值
                - 向量字段: chunk_dense、parent_chunk_dense、questions_dense、chunk_sparse 等
                           初始化为零向量或空字典，后续由调用方填充实际值

        输入样例：
            # 输入文档元数据字典
            doc_dict = {
                "hash_id": "doc_001_abc123def456",
                "chunk": "感冒是一种常见的上呼吸道感染性疾病，主要由病毒引起。常见症状包括发热、咳嗽、咽痛等。",
                "parent_chunk": "感冒（上呼吸道感染）是临床上最常见的呼吸系统疾病之一，包括普通感冒和流行性感冒。病毒是主要病原体，如鼻病毒、冠状病毒、流感病毒等。",
                "summary": "感冒是由病毒引起的上呼吸道感染，症状包括发热、咳嗽、咽痛。",
                "questions": [
                    "感冒的主要症状有哪些？",
                    "感冒是由什么引起的？",
                    "如何预防感冒？"
                ],
                "document": "感冒的完整文档内容...",
                "source": "medical_corpus_v1",
                "source_name": "医学知识库",
                "lt_doc_id": "lt_doc_12345",
                "chunk_id": 0,
            }

            origin_pk = "doc_001_abc123def456"
        输出样例：
            场景1： 主行 vector_id=-1
            # 调用
            row = _create_base_row(doc_dict, "doc_001_abc123def456", -1)

            # 输出
            row = {
                # === 主键和追踪字段 ===
                "pk": "doc_001_abc123def456",          # 主键，直接使用 origin_pk
                "origin_pk": "doc_001_abc123def456",   # 原始主键
                "vector_id": -1,                        # 标识为主行

                # === 基础字段（文本内容）===
                "chunk": "感冒是一种常见的上呼吸道感染性疾病，主要由病毒引起。常见症状包括发热、咳嗽、咽痛等。",
                "parent_chunk": "感冒（上呼吸道感染）是临床上最常见的呼吸系统疾病之一，包括普通感冒和流行性感冒。病毒是主要病原体，如鼻病毒、冠状病毒、流感病毒等。",
                "summary": "感冒是由病毒引起的上呼吸道感染，症状包括发热、咳嗽、咽痛。",
                "questions": "感冒的主要症状有哪些？, 感冒是由什么引起的？, 如何预防感冒？",  # list 转为逗号分隔字符串
                "document": "感冒的完整文档内容...",
                "source": "medical_corpus_v1",
                "source_name": "医学知识库",
                "lt_doc_id": "lt_doc_12345",
                "chunk_id": 0,
                "hash_id": "doc_001_abc123def456",

                # === 稠密向量字段（初始化为零向量，维度 1024）===
                "chunk_dense": [0.0] * 1024,           # 维度 1024，待填充实际向量
                "parent_chunk_dense": [0.0] * 1024,
                "summary_dense": [0.0] * 1024,         # 根据 index.yaml embed=false，实际可能不会创建
                "questions_dense": [0.0] * 1024,        # 展开行的占位符

                # === 稀疏向量字段（初始化为空字典）===
                "chunk_sparse": {},                     # 待填充 BM25 稀疏向量 {token_id: weight, ...}
            }
        场景2: 展开行（vector_id = 0，对应 questions[0]）
            # 调用
            row = _create_base_row(doc_dict, "doc_001_abc123def456", 0)

            # 输出
            row = {
                # === 主键和追踪字段 ===
                "pk": "doc_001_abc123def456_v0",       # 主键，追加 _v{vector_id}
                "origin_pk": "doc_001_abc123def456",   # 原始主键（用于回溯主行）
                "vector_id": 0,                         # 标识为 questions 列表的第 0 个元素

                # === 基础字段 ===
                "chunk": "感冒是一种常见的上呼吸道感染性疾病，主要由病毒引起。常见症状包括发热、咳嗽、咽痛等。",
                "parent_chunk": "感冒（上呼吸道感染）是临床上最常见的呼吸系统疾病之一，包括普通感冒和流行性感冒。病毒是主要病原体，如鼻病毒、冠状病毒、流感病毒等。",
                "summary": "感冒是由病毒引起的上呼吸道感染，症状包括发热、咳嗽、咽痛。",
                "questions": [                          # 保持原 list 类型，不转换为字符串
                    "感冒的主要症状有哪些？",
                    "感冒是由什么引起的？",
                    "如何预防感冒？"
                ],
                "document": "感冒的完整文档内容...",
                "source": "medical_corpus_v1",
                "source_name": "医学知识库",
                "lt_doc_id": "lt_doc_12345",
                "chunk_id": 0,
                "hash_id": "doc_001_abc123def456",

                # === 稠密向量字段 ===
                "chunk_dense": [0.0] * 1024,
                "parent_chunk_dense": [0.0] * 1024,
                "summary_dense": [0.0] * 1024,
                "questions_dense": [0.0] * 1024,        # 后续会填充为 "感冒的主要症状有哪些？" 的实际向量

                # === 稀疏向量字段 ===
                "chunk_sparse": {},                     # 稀疏向量通常只在主行填充
            }

        """
        row = {}

        # 主键字段
        # 若 Milvus 不自动生成ID（auto_id=False），则需要手动构造主键
        # 对于展开行（vector_id >= 0），主键格式为 "{origin_pk}_v{vector_id}" 以保证唯一性
        # 对于主行（vector_id = -1），主键直接使用 origin_pk
        if not self.config.milvus.auto_id:
            if vector_id >= 0:
                row["pk"] = f"{origin_pk}_v{vector_id}"
            else:
                row["pk"] = origin_pk

        # 追踪字段
        # origin_pk: 保存原始主键，用于查询时将同一文档的多条展开记录聚合
        # vector_id: 标识该行对应的向量元素，-1 表示主行，>=0 表示展开行的索引
        row["origin_pk"] = origin_pk
        row["vector_id"] = vector_id

        # 收集需要展开的 list 类型字段（如 questions）
        # 这些字段在配置中 type="list"，每个元素会生成一条独立的向量索引记录
        list_fields = []
        for field_name, field_config in self.config.dense_fields.items():
            if field_config.embed and field_config.type == "list":
                list_fields.append(field_name)

        # 复制基础字段到行中
        # 基础字段包含非向量的文本数据，如 chunk、parent_chunk、summary、document 等
        # 排除 pk、origin_pk、vector_id，因为它们已单独处理
        for field in self.config.base_fields:
            if field.name in ["pk", "origin_pk", "vector_id"]:
                continue
            field_value = doc_dict.get(field.name)
            if field_value is not None:
                # 对于主行（vector_id=-1），将 list 类型的字段转换为逗号分隔的字符串
                # 这样可以在主行中保留原始列表的完整信息，便于结果展示
                if vector_id == -1 and field.name in list_fields and isinstance(field_value, list):
                    row[field.name] = ", ".join(str(v) for v in field_value)
                else:
                    row[field.name] = field_value

        # 初始化所有稠密向量字段为全零向量
        # 向量维度由配置中的 dimension 指定（如 1024）
        # 全零向量作为占位符，后续由调用方根据需要填充实际向量值
        # 这种设计使得每一行都包含所有向量字段，便于统一处理和查询
        for field_config in self.config.dense_fields.values():
            if field_config.embed:
                row[field_config.index_field] = [0.0] * field_config.dimension

        # 初始化所有稀疏向量字段为空字典
        # 稀疏向量使用 Dict[int, float] 表示，key 为 token_id，value 为权重
        # 空字典表示该行没有对应的稀疏向量
        for field_config in self.config.sparse_fields.values():
            if field_config.embed:
                row[field_config.index_field] = {}

        return row

    @staticmethod
    def _compute_hash_id(document: Document) -> str:
        """计算文档哈希ID（简单实现）

        Args:
            document: 文档对象

        Returns:
            str: 哈希ID
        """
        import hashlib
        content = document.page_content + str(sorted(document.metadata.items()))
        return hashlib.md5(content.encode()).hexdigest()

    def vectorize_documents_batch(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """批量向量化文档

        Args:
            documents: 文档列表

        Returns:
            List[Dict[str, Any]]: 向量化后的行列表
        """
        all_rows = []
        for doc in documents:
            rows = self.vectorize_document(doc)
            all_rows.extend(rows)
        return all_rows
