"""Milvus Collection 管理器"""
import logging
from typing import Dict, Any, Optional

from pymilvus import MilvusClient, DataType, Collection, FunctionType, Function

from config import IndexConfig, BaseFieldConfig

logger = logging.getLogger(__name__)


class CollectionManager:
    """Milvus Collection 管理器"""

    def __init__(self, config: IndexConfig):
        """初始化 Collection 管理器

        Args:
            config: 索引配置
        """
        self.config = config
        self.milvus_config = config.milvus
        self.client = MilvusClient(
            uri=self.milvus_config.uri,
            token=self.milvus_config.token
        )

    def create_collection(self):
        """创建 Collection"""
        collection_name = self.milvus_config.collection_name

        # 如果配置了 drop_old，先删除旧集合
        if self.milvus_config.drop_old:
            if self.client.has_collection(collection_name=collection_name):
                logger.info(f"删除旧集合: {collection_name}")
                self.client.drop_collection(collection_name=collection_name)

        # 检查集合是否已存在
        if self.client.has_collection(collection_name=collection_name):
            logger.info(f"集合已存在: {collection_name}")
            return self.client

        # 创建 Schema
        schema = MilvusClient.create_schema(
            auto_id=self.milvus_config.auto_id,
            enable_dynamic_field=True,
        )

        # 添加主键字段
        self._add_primary_field(schema)

        # 添加基础字段（除了主键字段）
        # 基础字段包括：chunk, parent_chunk, summary, questions, document 等文本字段
        for field in self.config.base_fields:
            # 跳过以下字段，因为它们需要特殊处理：
            # 1. "pk": 主键字段，已经在 _add_primary_field() 中单独添加
            # 2. "origin_pk": 追踪字段，用于标识原始文档的主键（在多向量展开场景中，多个向量行可能对应同一个原始文档）
            # 3. "vector_id": 追踪字段，用于标识向量索引的序号（-1 表示主行，>=0 表示展开的向量行）
            # 这两个字段会在后续代码中单独添加，以便更好地控制它们的定义
            if field.name == "pk" or field.name in ["origin_pk", "vector_id"]:
                continue  # 这些字段特殊处理
            self._add_field(schema, field)

        # 添加追踪字段
        schema.add_field(
            field_name="origin_pk",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="vector_id",
            datatype=DataType.INT64
        )

        # 添加稠密向量字段
        for field_name, field_config in self.config.dense_fields.items():
            if field_config.embed:
                schema.add_field(
                    field_name=field_config.index_field,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=field_config.dimension
                )

        # 添加稀疏向量字段
        for field_name, field_config in self.config.sparse_fields.items():
            if field_config.embed:
                schema.add_field(
                    field_name=field_config.index_field,
                    datatype=DataType.SPARSE_FLOAT_VECTOR
                )

        # 添加 BM25 函数（如果使用 Milvus 内置 BM25）
        for field_name, field_config in self.config.sparse_fields.items():
            if field_config.embed and field_config.index_type == "BM25":
                bm25_fn = Function(
                    name=f"bm25_{field_name}_to_sparse",
                    function_type=FunctionType.BM25,
                    input_field_names=[field_name],
                    output_field_names=[field_config.index_field],
                )
                schema.add_function(bm25_fn)

        # 创建 Collection
        logger.info(f"创建集合: {collection_name}")
        self.client.create_collection(collection_name=collection_name, schema=schema)

        return self.client

    def _add_primary_field(self, schema):
        """添加主键字段到 Milvus Collection Schema

        主键是每个 Collection 中唯一标识文档的字段，用于：
        - 文档去重：相同主键的文档会被覆盖
        - 文档更新：通过主键定位并更新已有文档
        - 文档删除：通过主键精确删除文档
        - 追踪关联：在多向量展开场景中，origin_pk 追踪原始文档主键

        Args:
            schema: Milvus Schema 对象，用于定义 Collection 的字段结构

        主键类型选择逻辑:
        ---------------------
        1. auto_id=True (自动生成主键):
           - 使用 INT64 类型
           - Milvus 自动分配递增的整数 ID
           - 优点：简单，无需手动管理 ID
           - 缺点：
             * 无法进行文档去重（每次插入都是新 ID）
             * 无法进行文档更新（无法定位原文档）
             * 追踪原始文档需要额外的 hash_id 字段
           - 适用场景：临时数据、一次性导入、无需更新的场景

        2. auto_id=False (自定义主键，推荐):
           - 使用 VARCHAR 类型，最大长度 65535 字符
           - 由应用程序生成主键（通常是 hash_id）
           - 优点：
             * 支持文档去重（相同内容生成相同主键）
             * 支持文档更新（通过相同主键覆盖）
             * 追踪原始文档更直观
           - 缺点：
             * 需要额外代码生成唯一主键
             * VARCHAR 类型占用存储空间略大
           - 适用场景：长期存储、需要更新的知识库、生产环境

        配置示例 (index.yaml):
        ---------------------
        # 推荐配置
        milvus:
          auto_id: false  # 使用自定义主键

        # 数据结构中生成主键的方式:
        pk = hashlib.md5(chunk_content.encode()).hexdigest()
        """
        if self.milvus_config.auto_id:
            # 自动生成主键模式
            # - 数据类型：INT64 (64位整数)
            # - Milvus 自动分配递增 ID
            # - 适用于简单场景，但无法支持去重和更新
            schema.add_field(
                field_name="pk",
                datatype=DataType.INT64,
                is_primary=True
            )
        else:
            # 自定义主键模式（推荐）
            # - 数据类型：VARCHAR (可变长度字符串)
            # - 最大长度：65535 字符（64KB）
            # - 由应用程序生成唯一主键（通常是内容哈希）
            # - 支持文档去重、更新和删除
            schema.add_field(
                field_name="pk",
                datatype=DataType.VARCHAR,
                max_length=65535,
                is_primary=True
            )

    def _add_field(self, schema, field: BaseFieldConfig):
        """添加字段到 Schema

        Args:
            schema: Milvus Schema 对象
            field: 基础字段配置
        """
        if field.datatype == "VARCHAR":
            schema.add_field(
                field_name=field.name,
                datatype=DataType.VARCHAR,
                max_length=field.max_length,
                enable_analyzer=field.enable_analyzer
            )
        elif field.datatype == "INT64":
            schema.add_field(
                field_name=field.name,
                datatype=DataType.INT64
            )

    def build_index(self):
        """构建索引"""
        collection_name = self.milvus_config.collection_name
        index_params = self.client.prepare_index_params()

        # 稠密向量索引
        # 遍历所有稠密向量字段配置，为需要嵌入的字段创建索引
        # 稠密向量是连续的数值向量（如从 embedding 模型生成的 768/1536 维向量）
        # 用于语义相似度搜索，通过向量距离衡量语义相关性
        for field_name, field_config in self.config.dense_fields.items():
            if field_config.embed:
                # 为该字段添加向量索引
                # field_config.index_field: 索引字段名（如 "chunk_vector"）
                # field_config.index_type: 索引类型（如 "IVF_FLAT", "HNSW", "SCANN"）
                # field_config.metric_type: 距离度量类型（如 "IP" 内积, "L2" 欧氏距离）
                # field_config.index_params: 索引参数（如 {"nlist": 1024}）
                index_params.add_index(
                    field_name=field_config.index_field,
                    index_type=field_config.index_type,
                    index_name=f"{field_config.index_field}_index",
                    metric_type=field_config.metric_type,
                    params=field_config.index_params
                )

        # 稀疏向量索引
        for field_name, field_config in self.config.sparse_fields.items():
            if field_config.embed:
                index_params.add_index(
                    field_name=field_config.index_field,
                    index_type=field_config.index_type,
                    index_name=f"{field_config.index_field}_index",
                    metric_type=field_config.metric_type,
                    params=field_config.index_params
                )

        logger.info(f"创建索引: {collection_name}")
        self.client.create_index(collection_name=collection_name, index_params=index_params)

        # 加载 Collection
        logger.info(f"加载集合: {collection_name}")
        self.client.load_collection(collection_name)

    def drop_collection(self, collection_name: Optional[str] = None):
        """
        删除 Collection

        Args:
            collection_name: 集合名称，默认使用配置中的名称
        """
        name = collection_name or self.milvus_config.collection_name
        if self.client.has_collection(collection_name=name):
            logger.info(f"删除集合: {name}")
            self.client.drop_collection(collection_name=name)
