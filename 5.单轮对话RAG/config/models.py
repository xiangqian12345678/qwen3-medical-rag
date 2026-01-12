"""配置数据模型"""
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field


# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus向量数据库连接配置，用于管理Milvus连接和集合操作"""
    uri: str = "http://localhost:19530"  # Milvus服务器地址
    token: Optional[str] = None  # 认证令牌
    collection_name: str = "medical_knowledge"  # 集合名称
    drop_old: bool = False  # 是否删除旧集合
    auto_id: bool = False  # 是否自动生成主键


# =============================================================================
# 稠密向量字段配置
# =============================================================================
class DenseFieldConfig(BaseModel):
    """稠密向量字段配置，用于配置稠密向量的生成和索引参数"""
    provider: Literal['openai', 'ollama'] = 'ollama'  # 向量生成服务提供商
    model: str  # 嵌入模型名称
    base_url: Optional[str] = None  # 服务基础URL
    dimension: int = 1024  # 向量维度
    index_field: str  # 索引字段名
    index_type: Literal["HNSW", "IVF_FLAT", "IVF_PQ"] = "HNSW"  # 索引类型
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"ef": 64})  # 检索参数
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"  # 距离度量类型


# =============================================================================
# 稀疏向量字段配置
# =============================================================================
class SparseFieldConfig(BaseModel):
    """稀疏向量字段配置，用于配置稀疏向量（BM25）的生成和索引参数"""
    provider: Literal['self'] = 'self'  # 向量生成服务提供商
    vocab_path_or_name: str  # 词表路径或名称
    algorithm: str = "BM25"  # 稀疏向量算法
    domain_model: str = "medicine"  # 领域模型
    k1: float = 1.5  # BM25的k1参数
    b: float = 0.75  # BM25的b参数
    index_field: str  # 索引字段名
    index_type: Literal["SPARSE_INVERTED_INDEX"] = "SPARSE_INVERTED_INDEX"  # 索引类型
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"drop_ratio_search": 0.0})  # 检索参数
    metric_type: Literal["IP"] = "IP"  # 距离度量类型 公式: IP(x, y) = Σ(x_i * y_i)


# =============================================================================
# LLM 配置
# =============================================================================
class LLMConfig(BaseModel):
    """大语言模型配置，用于配置LLM的连接和生成参数"""
    provider: Literal['openai', 'ollama'] = 'ollama'  # LLM服务提供商
    model: str  # 模型名称
    base_url: Optional[str] = None  # 服务基础URL
    temperature: float = 0.1  # 生成温度
    max_tokens: Optional[int] = None  # 最大生成token数


# =============================================================================
# 数据字段配置
# =============================================================================
class DataConfig(BaseModel):
    """
    数据字段配置，用于定义数据在Milvus中的字段名称映射
    1.lt_doc_id (长文档ID)
        作用：
            将同一文档拆分的多个 chunks 关联起来
            在检索结果聚合时按文档分组
            追踪数据来源的原始文档
        代码：
            每 10 个 chunks 共享同一个 lt_doc_id
            构建 lt_doc_id（同一文档的 chunks 共享相同的 lt_doc_id）
            lt_doc_id = f"lt_doc_{chunk_id // 10}"
    2.chunk_id (文本块ID)
        作用
            标识单个 chunk 在文档中的顺序位置
            用于 chunk 去重和更新
            追踪数据处理流水线中的文本块
        代码
            数据标注阶段：从 1 开始递增，每次切分 chunk 后 chunk_id += 1
            位置 1: 3.索引构建/build_index.py:104-105
                chunk_id = data.get('chunk_id', 0)
                从输入数据中读取，如果不存在则默认为 0
            位置 2: 3.索引构建/ingest.py:132
                chunk_id = i  # 循环索引 i
            位置 3: 1.数据处理/1.4 数据标注/dataAnnotation.py:202, 238
                annotated['chunk_id'] = chunk_id  # 递增计数器
    3. hash_id (哈希ID)
        作用
            去重：相同内容生成相同 hash_id，插入时自动覆盖
            更新：通过相同 hash_id 更新已有文档
            唯一标识：作为文档在 Milvus 中的主键 (当 auto_id=False )
        计算方式与代码
            hash_id = hashlib.md5(chunk.encode('utf-8')).hexdigest()
    """
    chunk_field: str = "chunk"  # 文本块字段名
    parent_chunk_field: str = "parent_chunk"  # 父文本块字段名
    summary_field: str = "summary"  # 摘要字段名
    questions_field: str = "questions"  # 假设问题字段名
    source_field: str = "source"  # 来源字段名
    source_name_field: str = "source_name"  # 来源名称字段名
    lt_doc_id_field: str = "lt_doc_id"  # 长文档ID字段名
    chunk_id_field: str = "chunk_id"  # 文本块ID字段名
    hash_id_field: str = "hash_id"  # 哈希ID字段名


# =============================================================================
# RAG配置
# =============================================================================
class FieldSearchRequest(BaseModel):
    """单个字段检索配置，用于配置单个向量字段的检索参数"""
    anns_field: str = Field(description="向量检索字段")  # 向量检索字段名
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"  # 距离度量类型
    search_params: dict = Field(default_factory=lambda: {"ef": 64})  # 检索参数
    limit: int = Field(default=50, gt=0, le=500)  # 检索结果数量限制
    expr: str = ""  # 过滤表达式


class FusionConfig(BaseModel):
    """向量融合配置，用于配置多个向量检索结果的融合策略"""
    method: Literal["rrf", "weighted"] = "rrf"  # 融合方法：rrf(倒数排名融合)或weighted(加权融合)
    k: int = 60  # RRF的k参数
    weights: Dict[str, float] = Field(default_factory=lambda: {  # 各字段权重
        "chunk_dense": 0.35,
        "parent_chunk_dense": 0.35,
        "questions_dense": 0.20,
        "chunk_sparse": 0.10
    })


class RAGConfig(BaseModel):
    """基础RAG检索配置，用于配置检索增强生成的检索和融合参数"""
    default_fields: List[FieldSearchRequest] = Field(default_factory=list)  # 默认检索字段列表
    fusion: FusionConfig = Field(default_factory=FusionConfig)  # 融合配置
    output_fields: List[str] = Field(default_factory=lambda: [  # 输出字段列表
        "chunk", "parent_chunk", "summary", "questions",
        "source", "source_name", "lt_doc_id", "chunk_id", "hash_id"
    ])
    limit: int = 5  # 最终返回结果数量
    top_k: int = 50  # 每个字段检索的top_k数量


# =============================================================================
# 主配置
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置，统一管理所有子配置模块"""
    milvus: MilvusConfig  # Milvus配置
    dense_fields: Dict[str, DenseFieldConfig] = Field(default_factory=dict)  # 稠密向量字段配置
    sparse_fields: Dict[str, SparseFieldConfig] = Field(default_factory=dict)  # 稀疏向量字段配置
    llm: LLMConfig  # LLM配置
    data: DataConfig = Field(default_factory=DataConfig)  # 数据字段配置
    rag: RAGConfig = Field(default_factory=RAGConfig)  # RAG检索配置


# =============================================================================
# 检索请求模型
# =============================================================================
AnnsField = Literal["chunk_dense", "parent_chunk_dense", "summary_dense", "questions_dense", "chunk_sparse"]
OutputFields = Literal["pk", "chunk", "parent_chunk", "summary", "questions",
                       "source", "source_name", "lt_doc_id",
                       "chunk_id", "hash_id"]


class FusionSpec(BaseModel):
    """向量融合规范，用于API请求中的融合参数配置"""
    method: Literal["rrf", "weighted"] = "rrf"  # 融合方法
    k: Optional[int] = 60  # RRF的k参数
    weights: Optional[List[float]] = None  # 加权融合的权重列表


class SingleSearchRequest(BaseModel):
    """单个向量检索请求，用于API中的单字段检索参数配置"""
    anns_field: AnnsField = "chunk_dense"  # 向量检索字段名
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"  # 距离度量类型
    search_params: dict = Field(default_factory=lambda: {"ef": 64})  # 检索参数
    limit: int = Field(default=50, gt=0, le=500)  # 检索结果数量限制
    expr: str = ""  # 过滤表达式


class SearchRequest(BaseModel):
    """混合检索请求，用于API中的完整检索请求配置"""
    query: str = ""  # 查询文本
    collection_name: str = "medical_knowledge"  # 集合名称
    requests: List[SingleSearchRequest] = Field(default_factory=list)  # 检索请求列表
    output_fields: List[OutputFields] = Field(default_factory=list)  # 输出字段列表
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec)  # 融合配置
    top_k: int = 50  # 每个字段检索的top_k数量
    limit: int = 5  # 最终返回结果数量
