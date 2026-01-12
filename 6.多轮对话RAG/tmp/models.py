from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus配置

    用于配置Milvus向量数据库的连接参数和集合操作。

    字段说明:
    - uri: Milvus服务器地址，默认为本地部署地址"http://localhost:19530"
    - token: Milvus认证令牌，可选参数，用于安全连接
    - collection_name: 集合名称，用于存储医疗知识数据的Milvus集合
    - drop_old: 是否删除旧集合，True表示在数据导入前删除旧集合并重建，False表示保留旧数据
    - auto_id: 是否自动生成主键ID，True表示Milvus自动分配，False表示需手动指定ID
    """
    uri: str = "http://localhost:19530"  # Milvus服务器地址，默认本地localhost:19530端口
    token: Optional[str] = None  # 认证令牌，用于安全连接，默认为None（无需认证）
    collection_name: str = "medical_knowledge"  # 集合名称，存储医疗知识数据的集合标识
    drop_old: bool = False  # 是否删除旧集合，True=导入前删除重建，False=保留已有数据
    auto_id: bool = True  # 是否自动生成主键ID，True=系统自动分配，False=需手动指定


# =============================================================================
# 嵌入配置
# =============================================================================
class DenseConfig(BaseModel):
    """稠密向量配置"""
    provider: Literal['openai', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None
    dimension: int = 1024


class SparseConfig(BaseModel):
    """稀疏向量配置

    用于配置稀疏向量（BM25词袋模型）的生成参数和词汇表信息。

    字段说明:
    - provider: 稀疏向量生成方式，'self'表示自建BM25模型，'Milvus'表示使用Milvus内置的稀疏向量功能
    - vocab_path_or_name: 词汇表文件路径，支持.pkl.gz格式或预设模型名称
    - algorithm: 稀疏向量算法，目前支持"BM25"
    - domain_model: 领域模型类型，如"medicine"表示医疗领域
    - k1: BM25算法的k1参数，控制词频饱和度，默认1.5，范围通常在[1.2, 2.0]
    - b: BM25算法的b参数，控制文档长度归一化程度，默认0.75，范围通常在[0.0, 1.0]
    - build: 构建词汇表时的并行参数，workers为工作进程数，chunksize为每个进程处理的数据块大小
    """
    provider: Literal['self', 'Milvus'] = 'self'  # 稀疏向量生成方式: 'self'=自建BM25模型, 'Milvus'=使用Milvus内置功能
    vocab_path_or_name: str = "vocab.pkl.gz"  # 词汇表文件路径，如.pkl.gz格式或预设模型名称
    algorithm: str = "BM25"  # 稀疏向量算法类型，目前支持BM25
    domain_model: str = "medicine"  # 领域模型类型，如医疗(medicine)等
    k1: float = 1.5  # BM25参数k1: 控制词频饱和度，值越大对高频词的惩罚越小，范围[1.2, 2.0]
    b: float = 0.75  # BM25参数b: 控制文档长度归一化，值越大对长文档的惩罚越大，范围[0.0, 1.0]
    build: dict = {"workers": 8, "chunksize": 64}  # 构建参数: workers=并行工作进程数, chunksize=每进程处理数据块大小


# 更新嵌入配置，支持多向量
class EmbeddingConfig(BaseModel):
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
    proxy: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


# =============================================================================
# 数据配置
# =============================================================================
class DataConfig(BaseModel):
    """数据配置

    用于配置数据源的字段映射关系和默认值，支持问答(QA)和文献(literature)两种数据类型。

    字段说明:
    - summary_field: 概要字段（通常是问题），用于生成summary稠密向量
    - document_field: 文档字段（通常是答案），用于生成text稠密向量和text稀疏向量
    - source_field: 数据源类型字段，用于区分数据来源（如"qa"或"literature"）
    - source_name_field: 数据源名称字段，用于标记具体的数据源（如"huatuo"）
    - lt_doc_id_field: 长文本文档ID字段（文献数据独有），用于标识文档ID
    - chunk_id_field: 文档分块ID字段（文献数据独有），用于标识文档中的分块ID
    - default_source: 默认数据源类型，支持"qa"和"literature"两种
    - default_source_name: 默认数据源名称，用于QA数据的标识
    - default_lt_doc_id: 默认长文本文档ID，用于非文献数据
    - default_chunk_id: 默认分块ID，用于非文献数据
    """
    # 字段映射
    summary_field: str = "question"  # 概要字段名称，用于存储问题文本
    document_field: str = "answer"  # 文档字段名称，用于存储答案文本
    source_field: Optional[str] = None  # 数据源类型字段名（如source列名），用于区分数据来源
    source_name_field: Optional[str] = None  # 数据源名称字段名（如source_name列名），用于标记具体数据源

    ### 文档独有字段（文献数据使用） ###
    lt_doc_id_field: Optional[str] = None  # 长文本文档ID字段名，用于标识原始文献文档
    chunk_id_field: Optional[int] = None  # 文档分块ID字段名，用于标识文献中的分块顺序
    ### 文档独有字段 ###

    default_source: Optional[str] = "qa"  # 默认数据源类型: "qa"表示问答数据，"literature"表示文献数据
    default_source_name: Optional[str] = "huatuo"  # 默认数据源名称: 用于标识QA数据的来源（如华佗数据集）
    default_lt_doc_id: Optional[str] = ""  # 默认长文本文档ID: 用于非文献数据的占位
    default_chunk_id: Optional[int] = -1  # 默认分块ID: 用于非文献数据的占位


# =============================================================================
# 多轮RAG对话配置
# =============================================================================
class MultiDialogueRagConfig(BaseModel):
    """ 多轮对话关键配置
    
    该配置控制多轮RAG对话中的对话历史裁切、Token估算、调试等功能。
    主要用于在长对话场景下，通过动态裁切历史对话来控制上下文长度，
    避免超过LLM的输入限制。
    """
    # Token估算函数: "avg"使用平均值估算, "cl100k_base"使用tiktoken精确计算
    estimate_token_fun: str = "avg"

    # LLM的最大输入Token数限制,用于判断是否需要裁切对话历史
    llm_max_token: int = 1024

    # Token阈值系数: 当前总token数 > llm_max_token * max_token_threshold 时触发裁切
    # 1.1表示预留10%的余量作为宽松阈值
    max_token_threshold: float = 1.1  # 宽松阈值

    # 对话裁切比例: 每次裁切时，保留的对话轮数 = 当前轮数 / cut_dialogue_scale
    # 默认为2，即每次裁切砍掉一半的历史对话
    # 必须>=2，避免单次裁切过多
    cut_dialogue_scale: int = Field(default=2, ge=2, description="裁切一次砍一半，必须>=2")

    # Smith调试模式开关: 开启后会输出详细的Smith算法调试信息
    smith_debug: bool = False

    # 控制台调试模式开关: 开启后在控制台输出详细的对话处理日志
    console_debug: bool = False

    # 思维链模式开关: 开启后会在上下文中插入思考过程，增强推理能力但增加消耗
    thinking_in_context: bool = False


# =============================================================================
# Agent对话配置
# =============================================================================
class AgentConfig(BaseModel):
    """ 多轮对话关键配置 """
    # analysis 模式会拆解子目标分开多次检索，并验证是否符合事实,不符合事实需要重写检索
    # normal 模式下不会拆分子目标,只会重写查询后进行检索
    # fast 模式下重写查询检索后即返回,不进行验证事实
    mode: Literal["analysis", "fast", "normal"] = "analysis"
    max_attempts: int = 3  # 重复验证事实最大次数
    network_search_enabled: bool = True  # 是否启用联网搜索
    network_search_cnt: int = 10  # 开启联网搜索时，返回的数量
    auto_search_param: bool = True  # 是否开启确定搜索参数


# =============================================================================
# 更新主配置类
# =============================================================================
class AppConfig(BaseModel):
    """应用主配置 - 更新版"""
    milvus: MilvusConfig
    embedding: EmbeddingConfig  # 包含multi_vector配置
    llm: LLMConfig
    data: DataConfig
    multi_dialogue_rag: MultiDialogueRagConfig
    agent: AgentConfig


# =============================================================================
# 检索时需要传入的数据模型
# =============================================================================

AnnsField = Literal[
    "summary_dense", "text_dense", "text_sparse"
]

OutputFields = Literal[
    "pk", "text", "summary", "document", "source", "source_name",
    "lt_doc_id", "chunk_id", "summary_dense", "text_dense", "text_sparse"
]


class FusionSpec(BaseModel):
    method: Literal["rrf", "weighted"] = Field("rrf", description="向量融合策略")
    k: Optional[int] = Field(default=60, gt=0, le=200,
                             description="如果使用rrf融合策略,那么这个k值会影响结果")  # RRF常用k=60
    weights: Optional[List] = Field([0.3, 0.4, 0.3], description="如果使用weighted融合策略,那么这个weights会影响结果")


class SingleSearchRequest(BaseModel):
    anns_field: AnnsField = Field("summary_dense", description="向量检索字段")
    metric_type: Literal["COSINE", "IP"] = Field("COSINE", description="向量距离计算指标,除了稀疏向量,其余都用'COSINE'")
    search_params: dict = Field({"ef": 64},
                                description="如果是稀疏向量检索,那么应该指定drop_ratio_search,值为float,例如0.0,否则指定参数ef,值为int")
    limit: int = Field(default=50, gt=0, le=500, description="限制这个向量检索字段返回的多少条数据")
    expr: Optional[str] = Field("",
                                description="过滤不符合这个表达式的数据,例如当需要筛选数据源时,填入:'source == qa',一般不需要更改,除非用户指定")


class SearchRequest(BaseModel):
    query: str = Field("", description="查询文本")
    collection_name: str = Field(default="medical_knowledge", description="查询的collection,默认为'medical_knowledge'")
    requests: List[SingleSearchRequest] = Field(default_factory=lambda: [SingleSearchRequest()],
                                                description="多路向量查询的检索配置")
    output_fields: List[OutputFields] = Field(default_factory=lambda: ["text", "summary", "document"],
                                              description="最后输出的参考文档字段;text是由summary和document组合而来")
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec, description="向量融合策略")
    limit: int = Field(default=5, gt=0, le=10, description="经过融合排序之后,最终返回的数据量大小,请不要大于10篇")
