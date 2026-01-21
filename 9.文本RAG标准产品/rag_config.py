"""全局RAG配置数据模型"""
import sys
from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, Field

# 添加当前目录到 Python 路径（支持直接运行）
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 导入各子模块的配置类
from kgraph import Neo4jConfig
from milvus.embed_config import MilvusConfig, EmbeddingConfig


# =============================================================================
# LLM 配置
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
    max_attempts: int = 2
    auto_search_param: bool = True
    console_debug: bool = False
    max_ask_num: int = Field(default=5, gt=0, description="最大追问轮次")

    # 网络搜索配置参数
    network_search_enabled: bool = False
    network_search_cnt: int = 10


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
    neo4j: Neo4jConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    data: DataConfig
    agent: AgentConfig
    multi_dialogue_rag: MultiDialogueRagConfig
