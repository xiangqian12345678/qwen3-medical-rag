"""配置数据模型"""
from typing import Dict, List, Optional, Literal, Any
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
    provider: Literal['openai', 'ollama'] = 'ollama'
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
    build: dict = {"workers": 8, "chunksize": 64}


# =============================================================================
# 嵌入配置（多向量）
# =============================================================================
class EmbeddingConfig(BaseModel):
    """嵌入配置"""
    summary_dense: DenseConfig
    text_dense: DenseConfig
    text_sparse: SparseConfig


# =============================================================================
# LLM 配置
# =============================================================================
class LLMConfig(BaseModel):
    """LLM配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


# =============================================================================
# 数据配置
# =============================================================================
class DataConfig(BaseModel):
    """数据配置"""
    chunk_field: str = "chunk"
    parent_chunk_field: str = "parent_chunk"
    summary_field: str = "summary"
    questions_field: str = "questions"
    document_field: Optional[str] = None
    source_field: Optional[str] = "source"
    source_name_field: Optional[str] = "source_name"
    lt_doc_id_field: Optional[str] = "lt_doc_id"
    chunk_id_field: Optional[str] = "chunk_id"
    hash_id_field: Optional[str] = "hash_id"


# =============================================================================
# Agent配置
# =============================================================================
class AgentConfig(BaseModel):
    """Agent对话配置"""
    mode: Literal["analysis", "fast", "normal"] = "analysis"
    max_attempts: int = 2
    network_search_enabled: bool = False
    network_search_cnt: int = 10
    auto_search_param: bool = True
    console_debug: bool = False
    max_ask_num: int = 5  # 最大追问轮次


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
    cache_time: int = Field(default=60, ge=0, description="会话历史和摘要缓存超时时间（分钟）")
    summary_max_length: int = Field(default=500, ge=50, description="每次摘要生成的最大字符长度")
    summary_max_cache_count: int = Field(default=3, ge=1, description="缓存的摘要迭代次数上限")


# =============================================================================
# 主配置
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置"""
    milvus: MilvusConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    data: DataConfig
    agent: AgentConfig
    multi_dialogue_rag: MultiDialogueRagConfig


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
    search_params: dict = Field({"ef": 64})
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
