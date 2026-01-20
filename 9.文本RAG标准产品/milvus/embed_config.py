"""向量检索配置数据模型"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus向量数据库连接配置"""
    uri: str = "http://localhost:19530"
    token: Optional[str] = None
    collection_name: str = "medical_knowledge"
    drop_old: bool = False
    auto_id: bool = False


# =============================================================================
# 稠密向量字段配置
# =============================================================================
class DenseConfig(BaseModel):
    """稠密向量配置"""
    provider: Literal['openai', 'dashscope', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    dimension: int = 1024


# =============================================================================
# 稀疏向量字段配置
# =============================================================================
class SparseConfig(BaseModel):
    """稀疏向量配置（BM25）"""
    provider: Literal['self', 'Milvus'] = 'self'
    vocab_path_or_name: str = "vocab.pkl.gz"
    algorithm: str = "BM25"
    domain_model: str = "medicine"
    k1: float = 1.5
    b: float = 0.75
    build: Dict = {"workers": 8, "chunksize": 64}


# =============================================================================
# 嵌入配置（多向量）
# =============================================================================
class EmbeddingConfig(BaseModel):
    """嵌入配置"""
    summary_dense: DenseConfig
    text_dense: DenseConfig
    text_sparse: SparseConfig


# =============================================================================
# 检索请求模型
# =============================================================================
AnnsField = Literal["chunk_dense", "parent_chunk_dense", "questions_dense", "chunk_sparse"]

OutputFields = Literal[
    "pk", "chunk", "parent_chunk", "summary", "questions", "document", "source", "source_name",
    "lt_doc_id", "chunk_id", "hash_id"
]


class FusionSpec(BaseModel):
    """向量融合配置"""
    method: Literal["rrf", "weighted"] = "rrf"
    k: Optional[int] = Field(default=60, gt=0, le=200)
    weights: Optional[List] = Field([0.4, 0.4, 0.2])


class SingleSearchRequest(BaseModel):
    """单个检索请求"""
    anns_field: AnnsField = Field("chunk_dense")
    metric_type: Literal["COSINE", "IP"] = Field("COSINE")
    search_params: Dict = Field({"ef": 64})
    limit: int = Field(default=50, gt=0, le=500)
    expr: Optional[str] = Field("")


class SearchRequest(BaseModel):
    """检索请求"""
    query: str = Field("", description="查询文本")
    collection_name: str = Field(default="medical_knowledge")
    requests: List[SingleSearchRequest] = Field(
        default_factory=lambda: [SingleSearchRequest()]
    )
    output_fields: List[OutputFields] = Field(
        default_factory=lambda: ["chunk", "parent_chunk", "summary", "questions", "pk", "source", "source_name"]
    )
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec)
    limit: int = Field(default=5, gt=0, le=10)
