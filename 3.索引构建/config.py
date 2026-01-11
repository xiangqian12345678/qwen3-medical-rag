"""配置加载与模型定义"""
import logging
from typing import Dict, List, Optional, Literal, Any
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# =============================================================================
# 基础配置类
# =============================================================================

class MilvusConfig(BaseModel):
    """Milvus连接配置"""
    uri: str = "http://localhost:19530"  # Milvus服务地址
    token: Optional[str] = None  # 认证令牌
    collection_name: str = "medical_knowledge"  # 集合名称
    drop_old: bool = False  # 是否删除旧集合
    auto_id: bool = False  # 是否自动生成ID


class DenseFieldConfig(BaseModel):
    """稠密向量字段配置"""
    embed: bool = False
    type: Literal["str", "list"] = "str"
    dimension: int = 1024
    model: str = "bge-m3:latest"
    base_url: str = "http://localhost:11434"
    provider: Literal["openai", "ollama"] = "ollama"
    workers: int = 8
    index_field: str = ""
    index_type: Literal["HNSW", "IVF_FLAT", "IVF_PQ"] = "HNSW"
    index_params: Dict[str, Any] = Field(default_factory=lambda: {"M": 32, "efConstruction": 200})
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"ef": 64})
    metric_type: Literal["COSINE", "IP", "L2"] = "COSINE"


class SparseFieldConfig(BaseModel):
    """稀疏向量字段配置"""
    embed: bool = False
    vocab_path: str = "data/vocab.pkl.gz"
    algorithm: Literal["BM25"] = "BM25"
    k1: float = 1.5
    b: float = 0.75
    domain_model: str = "medicine"
    workers: int = 8
    index_field: str = ""
    index_type: Literal["SPARSE_INVERTED_INDEX"] = "SPARSE_INVERTED_INDEX"
    index_params: Dict[str, Any] = Field(default_factory=lambda: {"inverted_index_algo": "DAAT_MAXSCORE"})
    metric_type: Literal["IP", "BM25"] = "IP"


class BaseFieldConfig(BaseModel):
    """基础字段配置"""
    name: str  # 字段名称
    datatype: Literal["VARCHAR", "INT64"] = "VARCHAR"  # 字段数据类型
    max_length: int = 65535  # VARCHAR类型最大长度
    is_primary: bool = False  # 是否为主键
    enable_analyzer: bool = False  # 是否启用分析器


class FusionConfig(BaseModel):
    """检索融合配置"""
    method: Literal["rrf", "weighted"] = "rrf"
    k: int = 60
    weights: Dict[str, float] = Field(default_factory=dict)


class DefaultSearchConfig(BaseModel):
    """默认检索配置"""
    limit: int = 5
    top_k: int = 50
    output_fields: List[str] = Field(default_factory=list)


class IndexConfig(BaseModel):
    """索引主配置"""
    milvus: MilvusConfig
    dense_fields: Dict[str, DenseFieldConfig] = Field(default_factory=dict)
    sparse_fields: Dict[str, SparseFieldConfig] = Field(default_factory=dict)
    base_fields: List[BaseFieldConfig] = Field(default_factory=list)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    default_search: DefaultSearchConfig = Field(default_factory=DefaultSearchConfig)

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# 检索请求模型
# =============================================================================

AnnsField = Literal["chunk_dense", "parent_chunk_dense", "summary_dense", "questions_dense", "chunk_sparse"]
OutputFields = Literal["pk", "origin_pk", "vector_id", "chunk", "parent_chunk", "summary", 
                       "questions", "document", "source", "source_name", "lt_doc_id", 
                       "chunk_id", "hash_id", "chunk_dense", "parent_chunk_dense", 
                       "summary_dense", "questions_dense", "chunk_sparse"]


class FusionSpec(BaseModel):
    """向量融合规范"""
    method: Literal["rrf", "weighted"] = Field(default="rrf", description="向量融合策略")
    k: Optional[int] = Field(default=60, gt=0, le=200, description="RRF的k值")
    weights: Optional[Dict[str, float]] = Field(default=None, description="加权融合的权重")


class SingleSearchRequest(BaseModel):
    """单个向量检索请求"""
    anns_field: AnnsField = Field(description="向量检索字段")
    metric_type: Literal["COSINE", "IP", "L2"] = Field(default="COSINE", description="距离计算指标")
    search_params: Dict[str, Any] = Field(default_factory=lambda: {"ef": 64}, description="检索参数")
    limit: int = Field(default=50, gt=0, le=500, description="返回文档数量限制")
    expr: Optional[str] = Field(default="", description="过滤表达式")


class SearchRequest(BaseModel):
    """混合检索请求"""
    query: str = Field(description="查询文本")
    collection_name: str = Field(default="medical_knowledge", description="集合名称")
    requests: List[SingleSearchRequest] = Field(default_factory=list, description="多路向量查询配置")
    output_fields: List[OutputFields] = Field(default_factory=list, description="输出字段")
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec, description="向量融合策略")
    top_k: int = Field(default=50, gt=0, le=500, description="融合时每个子请求的返回数量")
    limit: int = Field(default=5, gt=0, le=10, description="最终返回数量")


# =============================================================================
# 配置加载器
# =============================================================================

class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_root = Path(__file__).parent
            self.config_path = config_root / "index.yaml"
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        self._config = IndexConfig(**raw)

    @property
    def config(self) -> IndexConfig:
        """获取当前配置"""
        return self._config

    @property
    def as_dict(self) -> dict:
        """返回配置的字典形式"""
        return self._config.model_dump()

    def save(self, save_path: Optional[str] = None):
        """保存配置到文件"""
        path = Path(save_path) if save_path else self.config_path
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._config.model_dump(), f, allow_unicode=True, sort_keys=False)
