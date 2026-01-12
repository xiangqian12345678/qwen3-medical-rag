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
    uri: str = "http://localhost:19530"
    token: Optional[str] = None
    collection_name: str = "medical_knowledge"
    drop_old: bool = False
    auto_id: bool = True


# =============================================================================
# 稠密向量字段配置
# =============================================================================
class DenseFieldConfig(BaseModel):
    """稠密向量字段配置，用于配置稠密向量的生成和索引参数"""
    provider: Literal['openai', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    dimension: int = 1024
    index_field: str  # 索引字段名
    index_type: Literal["HNSW", "IVF_FLAT", "IVF_PQ"] = "HNSW"
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"ef": 64})
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"


# =============================================================================
# 稀疏向量字段配置
# =============================================================================
class SparseFieldConfig(BaseModel):
    """稀疏向量字段配置，用于配置稀疏向量（BM25）的生成和索引参数"""
    provider: Literal['self'] = 'self'
    vocab_path_or_name: str  # 词表路径或名称
    algorithm: str = "BM25"
    domain_model: str = "medicine"
    k1: float = 1.5
    b: float = 0.75
    index_field: str  # 索引字段名
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
    api_key: Optional[str] = None
    proxy: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


# =============================================================================
# 数据字段配置
# =============================================================================
class DataConfig(BaseModel):
    """数据字段配置，用于定义数据在Milvus中的字段名称映射"""
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
# 多轮对话RAG配置
# =============================================================================
class MultiDialogueRagConfig(BaseModel):
    """多轮对话RAG配置"""
    estimate_token_fun: str = "avg"
    llm_max_token: int = 1024
    max_token_threshold: float = 1.01
    cut_dialogue_scale: int = Field(default=2, ge=2)
    smith_debug: bool = False
    console_debug: bool = False
    thinking_in_context: bool = False


# =============================================================================
# Agent配置
# =============================================================================
class AgentConfig(BaseModel):
    """Agent配置"""
    mode: Literal["analysis", "fast", "normal"] = "analysis"
    max_attempts: int = 3
    network_search_enabled: bool = True
    network_search_cnt: int = 10
    auto_search_param: bool = True


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
    multi_dialogue_rag: MultiDialogueRagConfig
    agent: AgentConfig


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
