"""配置数据模型"""
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field


# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus向量数据库连接配置"""
    uri: str = "http://localhost:19530"  # Milvus服务地址
    token: Optional[str] = None  # 认证令牌
    collection_name: str = "medical_knowledge"  # 集合名称
    drop_old: bool = False  # 是否删除旧集合
    auto_id: bool = False  # 是否自动生成ID


# =============================================================================
# 稠密向量字段配置
# =============================================================================
class DenseConfig(BaseModel):
    """稠密向量配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'  # 向量服务提供商
    model: str  # 模型名称
    base_url: Optional[str] = None  # API基础URL
    api_key: Optional[str] = None  # API密钥
    dimension: int = 1024  # 向量维度


# =============================================================================
# 稀疏向量字段配置
# =============================================================================
class SparseConfig(BaseModel):
    """稀疏向量配置（BM25）"""
    provider: Literal['self', 'Milvus'] = 'self'  # 向量服务提供商
    vocab_path_or_name: str = "vocab.pkl.gz"  # 词表文件路径
    algorithm: str = "BM25"  # 算法类型
    domain_model: str = "medicine"  # 领域模型
    k1: float = 1.5  # BM25参数k1
    b: float = 0.75  # BM25参数b
    build: dict = {"workers": 8, "chunksize": 64}  # 构建参数


# =============================================================================
# 嵌入配置（多向量）
# =============================================================================
class EmbeddingConfig(BaseModel):
    """嵌入配置"""
    summary_dense: DenseConfig  # 摘要稠密向量配置
    text_dense: DenseConfig  # 文本稠密向量配置
    text_sparse: SparseConfig  # 文本稀疏向量配置


# =============================================================================
# LLM 配置
# =============================================================================
class LLMConfig(BaseModel):
    """LLM配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'  # 服务提供商
    model: str  # 模型名称
    base_url: Optional[str] = None  # API基础URL
    api_key: Optional[str] = None  # API密钥
    temperature: float = 0.1  # 温度参数
    max_tokens: Optional[int] = None  # 最大token数


# =============================================================================
# 数据配置
# =============================================================================
class DataConfig(BaseModel):
    """数据配置"""
    chunk_field: str = "chunk"  # 文本块字段名
    parent_chunk_field: str = "parent_chunk"  # 父文本块字段名
    summary_field: str = "summary"  # 摘要字段名
    questions_field: str = "questions"  # 问题字段名
    document_field: Optional[str] = None  # 文档字段名
    source_field: Optional[str] = "source"  # 来源字段名
    source_name_field: Optional[str] = "source_name"  # 来源名称字段名
    lt_doc_id_field: Optional[str] = "lt_doc_id"  # 文档ID字段名
    chunk_id_field: Optional[str] = "chunk_id"  # 文本块ID字段名
    hash_id_field: Optional[str] = "hash_id"  # 哈希ID字段名


# =============================================================================
# Agent配置
# =============================================================================
class AgentConfig(BaseModel):
    """Agent对话配置"""
    # 运行模式: analysis-深度分析(准确最高), fast-快速响应(速度最快), normal-均衡模式
    mode: Literal["analysis", "fast", "normal"] = "analysis"
    max_attempts: int = 2  # 最大尝试次数
    network_search_enabled: bool = False  # 是否启用网络搜索
    network_search_cnt: int = 10  # 网络搜索结果数量
    auto_search_param: bool = True  # 自动搜索参数
    console_debug: bool = False  # 控制台调试


# =============================================================================
# 主配置
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置"""
    milvus: MilvusConfig  # Milvus配置
    embedding: EmbeddingConfig  # 嵌入配置
    llm: LLMConfig  # LLM配置
    data: DataConfig  # 数据配置
    agent: AgentConfig  # Agent配置


# =============================================================================
# 检索请求模型
# =============================================================================
AnnsField = Literal["chunk_dense", "parent_chunk_dense", "questions_dense", "chunk_sparse"]  # 向量字段类型

OutputFields = Literal[  # 输出字段类型
    "pk", "chunk", "parent_chunk", "summary", "questions", "document", "source", "source_name",
    "lt_doc_id", "chunk_id", "chunk_dense", "parent_chunk_dense", "questions_dense", "chunk_sparse",
    "question", "answer"
]


class FusionSpec(BaseModel):
    """向量融合配置"""
    method: Literal["rrf", "weighted"] = "rrf"  # 融合方法
    k: Optional[int] = Field(default=60, gt=0, le=200)  # RRF参数k
    weights: Optional[List] = Field([0.4, 0.4, 0.2])  # 权重列表


class SingleSearchRequest(BaseModel):
    """单个检索请求"""
    anns_field: AnnsField = Field("chunk_dense")  # 向量字段
    metric_type: Literal["COSINE", "IP"] = Field("COSINE")  # 相似度度量类型
    search_params: dict = Field({"ef": 64})  # 搜索参数
    limit: int = Field(default=50, gt=0, le=500)  # 返回结果数
    expr: Optional[str] = Field("")  # 过滤表达式


class SearchRequest(BaseModel):
    """检索请求"""
    query: str = Field("", description="查询文本")  # 查询文本
    collection_name: str = Field(default="medical_knowledge")  # 集合名称
    requests: List[SingleSearchRequest] = Field(  # 检索请求列表
        default_factory=lambda: [SingleSearchRequest()]
    )
    output_fields: List[OutputFields] = Field(  # 输出字段列表
        default_factory=lambda: ["chunk", "parent_chunk", "summary", "questions", "pk", "source", "source_name"]
    )
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec)  # 向量融合配置
    limit: int = Field(default=5, gt=0, le=10)  # 最终返回结果数
