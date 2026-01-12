"""配置数据模型"""
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field


# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus配置

    用于配置Milvus向量数据库的连接参数和集合操作。
    """
    uri: str = "http://localhost:19530"  # Milvus服务地址
    token: Optional[str] = None  # 认证令牌
    collection_name: str = "medical_knowledge"  # 集合名称
    drop_old: bool = False  # 是否删除旧集合
    auto_id: bool = True  # 是否自动生成ID


# =============================================================================
# 稠密向量字段配置
# =============================================================================
class DenseFieldConfig(BaseModel):
    """稠密向量字段配置，用于配置稠密向量的生成和索引参数"""
    provider: Literal['openai', 'ollama'] = 'ollama'  # 向量服务提供商
    model: str  # 模型名称
    base_url: Optional[str] = None  # 服务基础URL
    dimension: int = 1024  # 向量维度
    index_field: str  # 索引字段名
    index_type: Literal["HNSW", "IVF_FLAT", "IVF_PQ"] = "HNSW"  # 索引类型
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"ef": 64})  # 搜索参数
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"  # 距离度量类型


# =============================================================================
# 稀疏向量字段配置
# =============================================================================
class SparseFieldConfig(BaseModel):
    """稀疏向量字段配置，用于配置稀疏向量（BM25）的生成和索引参数"""
    provider: Literal['self'] = 'self'  # 向量服务提供商
    vocab_path_or_name: str  # 词表路径或名称
    algorithm: str = "BM25"  # 算法类型
    domain_model: str = "medicine"  # 领域模型
    k1: float = 1.5  # BM25参数k1
    b: float = 0.75  # BM25参数b
    index_field: str  # 索引字段名
    index_type: Literal["SPARSE_INVERTED_INDEX"] = "SPARSE_INVERTED_INDEX"  # 索引类型
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"drop_ratio_search": 0.0})  # 搜索参数
    metric_type: Literal["IP"] = "IP"  # 距离度量类型


# =============================================================================
# LLM 配置
# =============================================================================
class LLMConfig(BaseModel):
    """LLM配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'  # LLM服务提供商
    model: str  # 模型名称
    base_url: Optional[str] = None  # 服务基础URL
    api_key: Optional[str] = None  # API密钥
    proxy: Optional[str] = None  # 代理地址
    temperature: float = 0.1  # 温度参数
    max_tokens: Optional[int] = None  # 最大生成token数


# =============================================================================
# 数据字段配置
# =============================================================================
class DataConfig(BaseModel):
    """数据字段配置，用于定义数据在Milvus中的字段名称映射"""
    chunk_field: str = "chunk"  # 文本块字段名
    parent_chunk_field: str = "parent_chunk"  # 父文本块字段名
    summary_field: str = "summary"  # 摘要字段名
    questions_field: str = "questions"  # 问题字段名
    document_field: str = "document"  # 文档字段名
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
        "chunk_dense": 0.40,
        "parent_chunk_dense": 0.40,
        "questions_dense": 0.15,
        "chunk_sparse": 0.05
    })


class RAGConfig(BaseModel):
    """基础RAG检索配置，用于配置检索增强生成的检索和融合参数"""
    default_fields: List[FieldSearchRequest] = Field(default_factory=list)  # 默认检索字段列表
    fusion: FusionConfig = Field(default_factory=FusionConfig)  # 融合配置
    output_fields: List[str] = Field(default_factory=lambda: [  # 输出字段列表
        "chunk", "parent_chunk", "summary", "questions", "document",
        "source", "source_name", "lt_doc_id", "chunk_id", "hash_id"
    ])
    limit: int = 5  # 最终返回结果数量
    top_k: int = 50  # 每个字段检索的top_k数量


# =============================================================================
# 多轮对话RAG配置
# =============================================================================
class MultiDialogueRagConfig(BaseModel):
    """多轮对话RAG配置"""
    estimate_token_fun: str = "avg"  # token估算方法
    llm_max_token: int = 1024  # LLM最大token数
    max_token_threshold: float = 1.01  # 最大token阈值
    cut_dialogue_scale: int = Field(default=2, ge=2)  # 对话裁剪轮数
    smith_debug: bool = False  # Smith调试开关
    console_debug: bool = False  # 控制台调试开关
    thinking_in_context: bool = False  # 上下文思考开关
    cache_time: int = Field(default=60, ge=0, description="会话历史和摘要缓存超时时间（分钟），0表示不超时")  # 缓存超时时间


# =============================================================================
# 主配置
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置"""
    milvus: MilvusConfig  # Milvus配置
    dense_fields: Dict[str, DenseFieldConfig] = Field(default_factory=dict)  # 稠密向量字段配置
    sparse_fields: Dict[str, SparseFieldConfig] = Field(default_factory=dict)  # 稀疏向量字段配置
    llm: LLMConfig  # LLM配置
    data: DataConfig = Field(default_factory=DataConfig)  # 数据字段配置
    rag: RAGConfig = Field(default_factory=RAGConfig)  # RAG检索配置
    multi_dialogue_rag: MultiDialogueRagConfig = Field(default_factory=MultiDialogueRagConfig)  # 多轮对话RAG配置


# =============================================================================
# 检索请求模型
# =============================================================================
AnnsField = Literal["chunk_dense", "parent_chunk_dense", "questions_dense", "chunk_sparse"]

OutputFields = Literal[
    "pk", "chunk", "parent_chunk", "summary", "questions",
    "document", "source", "source_name", "lt_doc_id",
    "chunk_id", "hash_id"
]


class FusionSpec(BaseModel):
    """向量融合规范"""
    method: Literal["rrf", "weighted"] = "rrf"  # 融合方法
    k: Optional[int] = 60  # RRF参数k
    weights: Optional[List[float]] = None  # 加权融合权重


class SingleSearchRequest(BaseModel):
    """单个向量检索请求"""
    anns_field: AnnsField = "chunk_dense"  # 向量索引字段
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"  # 距离度量类型
    search_params: dict = Field(default_factory=lambda: {"ef": 64})  # 搜索参数
    limit: int = Field(default=50, gt=0, le=500)  # 返回结果数
    expr: str = ""  # 过滤表达式


class SearchRequest(BaseModel):
    """混合检索请求"""
    query: str = ""  # 查询文本
    collection_name: str = "medical_knowledge"  # 集合名称
    requests: List[SingleSearchRequest] = Field(default_factory=list)  # 检索请求列表
    output_fields: List[OutputFields] = Field(default_factory=list)  # 输出字段列表
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec)  # 融合规范
    top_k: int = 50  # 检索返回数
    limit: int = 5  # 最终返回数
