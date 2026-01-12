"""配置数据模型"""
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field


# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus连接配置"""
    uri: str = "http://localhost:19530"
    token: Optional[str] = None
    collection_name: str = "medical_knowledge"
    drop_old: bool = False
    auto_id: bool = False


# =============================================================================
# 稠密向量字段配置
# =============================================================================
class DenseFieldConfig(BaseModel):
    """稠密向量字段配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    dimension: int = 1024
    index_field: str
    index_type: Literal["HNSW", "IVF_FLAT", "IVF_PQ"] = "HNSW"
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"ef": 64})
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"


# =============================================================================
# 稀疏向量字段配置
# =============================================================================
class SparseFieldConfig(BaseModel):
    """稀疏向量字段配置"""
    provider: Literal['self'] = 'self'
    vocab_path_or_name: str
    algorithm: str = "BM25"
    domain_model: str = "medicine"
    k1: float = 1.5
    b: float = 0.75
    index_field: str
    index_type: Literal["SPARSE_INVERTED_INDEX"] = "SPARSE_INVERTED_INDEX"
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"drop_ratio_search": 0.0})
    metric_type: Literal["IP"] = "IP"


# =============================================================================
# LLM 配置
# =============================================================================
class LLMConfig(BaseModel):
    """LLM配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


# =============================================================================
# 数据字段配置
# =============================================================================
class DataConfig(BaseModel):
    """数据字段配置"""
    chunk_field: str = "chunk"
    parent_chunk_field: str = "parent_chunk"
    summary_field: str = "summary"
    questions_field: str = "questions"
    document_field: str = "document"
    source_field: str = "source"
    source_name_field: str = "source_name"
    lt_doc_id_field: str = "lt_doc_id"
    chunk_id_field: str = "chunk_id"
    hash_id_field: str = "hash_id"


# =============================================================================
# RAG配置
# =============================================================================
class FieldSearchRequest(BaseModel):
    """单个字段检索配置"""
    anns_field: str = Field(description="向量检索字段")
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"
    search_params: dict = Field(default_factory=lambda: {"ef": 64})
    limit: int = Field(default=50, gt=0, le=500)
    expr: str = ""


class FusionConfig(BaseModel):
    """向量融合配置"""
    method: Literal["rrf", "weighted"] = "rrf"
    k: int = 60
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "chunk_dense": 0.35,
        "parent_chunk_dense": 0.35,
        "questions_dense": 0.20,
        "chunk_sparse": 0.10
    })


class RAGConfig(BaseModel):
    """基础RAG检索配置"""
    default_fields: List[FieldSearchRequest] = Field(default_factory=list)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    output_fields: List[str] = Field(default_factory=lambda: [
        "chunk", "parent_chunk", "summary", "questions", "document",
        "source", "source_name", "lt_doc_id", "chunk_id", "hash_id"
    ])
    limit: int = 5
    top_k: int = 50


# =============================================================================
# 主配置
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置"""
    milvus: MilvusConfig
    dense_fields: Dict[str, DenseFieldConfig] = Field(default_factory=dict)
    sparse_fields: Dict[str, SparseFieldConfig] = Field(default_factory=dict)
    llm: LLMConfig
    data: DataConfig = Field(default_factory=DataConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)


# =============================================================================
# 检索请求模型
# =============================================================================
AnnsField = Literal["chunk_dense", "parent_chunk_dense", "summary_dense", "questions_dense", "chunk_sparse"]
OutputFields = Literal["pk", "chunk", "parent_chunk", "summary", "questions",
                       "document", "source", "source_name", "lt_doc_id",
                       "chunk_id", "hash_id"]


class FusionSpec(BaseModel):
    """向量融合规范"""
    method: Literal["rrf", "weighted"] = "rrf"
    k: Optional[int] = 60
    weights: Optional[List[float]] = None


class SingleSearchRequest(BaseModel):
    """单个向量检索请求"""
    anns_field: AnnsField = "chunk_dense"
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"
    search_params: dict = Field(default_factory=lambda: {"ef": 64})
    limit: int = Field(default=50, gt=0, le=500)
    expr: str = ""


class SearchRequest(BaseModel):
    """混合检索请求"""
    query: str = ""
    collection_name: str = "medical_knowledge"
    requests: List[SingleSearchRequest] = Field(default_factory=list)
    output_fields: List[OutputFields] = Field(default_factory=list)
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec)
    top_k: int = 50
    limit: int = 5
