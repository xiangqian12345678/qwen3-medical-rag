"""向量检索配置数据模型 - 与索引构建完全对齐"""
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field


# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus向量数据库连接配置"""
    uri: str = Field(default="http://localhost:19530", description="Milvus服务器地址")
    token: Optional[str] = Field(default=None, description="Milvus认证令牌")
    collection_name: str = Field(default="medical_knowledge", description="集合名称")
    drop_old: bool = Field(default=False, description="是否删除旧集合重新创建")
    auto_id: bool = Field(default=False, description="是否自动生成主键ID")

# =============================================================================
# 稠密向量字段配置
# =============================================================================
class DenseFieldConfig(BaseModel):
    """稠密向量字段配置"""
    embed: bool = Field(default=False, description="是否启用嵌入生成")
    type: Literal["str", "list"] = Field(default="str", description="字段类型：str-字符串文本，list-已有向量")
    workers: int = Field(default=8, description="并发处理线程数")
    index_field: str = Field(default="", description="索引字段名称")
    index_type: Literal["HNSW", "IVF_FLAT", "IVF_PQ"] = Field(default="HNSW", description="向量索引类型")
    index_params: Dict[str, Any] = Field(default_factory=lambda: {"M": 32, "efConstruction": 200}, description="索引构建参数")
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"ef": 64}, description="向量检索参数")
    metric_type: Literal["COSINE", "IP", "L2"] = Field(default="COSINE", description="距离度量类型：COSINE-余弦相似度，IP-内积，L2-欧式距离")


# =============================================================================
# 稀疏向量字段配置
# =============================================================================
class SparseFieldConfig(BaseModel):
    """稀疏向量字段配置（BM25）"""
    embed: bool = Field(default=False, description="是否启用稀疏向量嵌入")
    vocab_path: str = Field(default="vocab.pkl.gz", description="BM25词表文件路径")
    algorithm: Literal["BM25"] = Field(default="BM25", description="稀疏向量算法类型")
    k1: float = Field(default=1.5, description="BM25参数k1，控制词频饱和度")
    b: float = Field(default=0.75, description="BM25参数b，控制文档长度归一化程度")
    domain_model: str = Field(default="medicine", description="领域模型名称")
    workers: int = Field(default=8, description="并发处理线程数")
    index_field: str = Field(default="", description="索引字段名称")
    index_type: Literal["SPARSE_INVERTED_INDEX"] = Field(default="SPARSE_INVERTED_INDEX", description="稀疏向量索引类型")
    index_params: Dict[str, Any] = Field(default_factory=lambda: {"inverted_index_algo": "DAAT_MAXSCORE"}, description="索引构建参数")
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"ef": 64}, description="检索参数")
    metric_type: Literal["IP", "BM25"] = Field(default="IP", description="稀疏向量度量类型")


# =============================================================================
# 基础字段配置
# =============================================================================
class BaseFieldConfig(BaseModel):
    """基础字段配置"""
    name: str = Field(description="字段名称")
    datatype: Literal["VARCHAR", "INT64"] = Field(default="VARCHAR", description="字段数据类型")
    max_length: int = Field(default=65535, description="字符串字段最大长度")
    is_primary: bool = Field(default=False, description="是否为主键字段")
    enable_analyzer: bool = Field(default=False, description="是否启用文本分词器")


# =============================================================================
# 检索融合配置
# =============================================================================
class FusionConfig(BaseModel):
    """检索融合配置"""
    method: Literal["rrf", "weighted"] = Field(default="rrf", description="融合方法：rrf-倒数排名融合，weighted-加权融合")
    k: int = Field(default=60, description="RRF融合的k值参数")
    weights: Dict[str, float] = Field(default_factory=dict, description="加权融合的权重映射，键为字段名")


# =============================================================================
# 默认检索配置
# =============================================================================
class DefaultSearchConfig(BaseModel):
    """默认检索配置"""
    limit: int = Field(default=5, description="最终返回的文档数量")
    top_k: int = Field(default=50, description="向量检索时的返回数量")
    output_fields: List[str] = Field(default_factory=list, description="需要输出的字段列表")


# =============================================================================
# 嵌入配置（主配置）
# =============================================================================
class EmbedConfig(BaseModel):
    """嵌入配置主模型"""
    milvus: MilvusConfig = Field(description="Milvus数据库连接配置")
    dense_fields: Dict[str, DenseFieldConfig] = Field(default_factory=dict, description="稠密向量字段配置字典")
    sparse_fields: Dict[str, SparseFieldConfig] = Field(default_factory=dict, description="稀疏向量字段配置字典")
    fusion: FusionConfig = Field(default_factory=FusionConfig, description="多路检索融合配置")
    default_search: DefaultSearchConfig = Field(default_factory=DefaultSearchConfig, description="默认检索参数配置")

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# 检索请求模型
# =============================================================================
# 下面两个类型没有用，将来可以删除
AnnsField = Literal["chunk_dense", "parent_chunk_dense", "summary_dense", "questions_dense", "chunk_sparse"]

# OutputFields = Literal[
#     "pk", "origin_pk", "vector_id", "chunk", "parent_chunk", "summary", "questions",
#     "document", "source", "source_name", "lt_doc_id", "chunk_id", "hash_id"
# ]


class FusionSpec(BaseModel):
    """向量融合配置"""
    method: Literal["rrf", "weighted"] = Field(default="rrf", description="向量融合策略")
    k: Optional[int] = Field(default=60, gt=0, le=200, description="RRF的k值")
    weights: Optional[Dict[str, float]] = Field(default=None, description="加权融合的权重")


class SingleSearchRequest(BaseModel):
    """单个检索请求"""
    anns_field: AnnsField = Field(description="向量检索字段")
    metric_type: Literal["COSINE", "IP", "L2"] = Field(default="COSINE", description="距离计算指标")
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"ef": 64}, description="检索参数")
    limit: int = Field(default=50, gt=0, le=500, description="返回文档数量限制")
    expr: Optional[str] = Field(default="", description="过滤表达式")


class SearchRequest(BaseModel):
    """检索请求"""
    query: str = Field(description="查询文本")
    collection_name: str = Field(default="medical_knowledge", description="集合名称")
    requests: List[SingleSearchRequest] = Field(default_factory=list, description="多路向量查询配置")
    output_fields: List[str] = Field(default_factory=list, description="输出字段")
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec, description="向量融合策略")
    top_k: int = Field(default=50, gt=0, le=500, description="融合时每个子请求的返回数量")
    limit: int = Field(default=5, gt=0, le=10, description="最终返回数量")
