"""全局RAG配置数据模型"""
from typing import Optional, Literal

from pydantic import BaseModel, Field


# =============================================================================
# LLM配置
# =============================================================================
class LLMConfig(BaseModel):
    """LLM配置"""
    provider: Literal['openai', 'dashscope', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


# =============================================================================
# Agent配置
# =============================================================================
class AgentConfig(BaseModel):
    """Agent对话配置"""
    max_attempts: int = 2
    console_debug: bool = False
    max_ask_num: int = Field(default=5, gt=0, description="最大追问轮次")

    # 网络搜索配置参数
    network_search_enabled: bool = False
    network_search_cnt: int = 10

    # 知识图谱配置参数
    kgraph_search_enabled: bool = False
    kgraph_search_cnt: int = 10

    # Query强化
    query_intent_enabled: bool = False
    query_rewrite_enabled: bool = False
    query_refine_enabled: bool = False

    # 召回强化
    generate_multi_queries_enabled: bool = True
    generate_sub_queries_enabled: bool = True
    generate_superordinate_query_enabled: bool = True
    generate_hypothetical_answer_enabled: bool = True

    # 过滤
    filter_low_correction_content_enabled: bool = True
    filter_low_correction_doc_llm_enabled: bool = True
    filter_low_correction_doc_embeddings_enabled: bool = True
    low_correction_threshold: float = 0.65
    filter_redundant_doc_embeddings_enabled: bool = True
    redundant_threshold: float = 0.95

    # 排序
    sort_docs_cross_encoder_enabled: bool = True
    sort_docs_by_loss_of_location_enabled: bool = True


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
# rag配置
# =============================================================================
class RAGConfig(BaseModel):
    """应用主配置"""
    llm: LLMConfig
    agent: AgentConfig
    multi_dialogue_rag: MultiDialogueRagConfig
