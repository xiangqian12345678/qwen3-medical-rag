"""知识图谱配置数据模型"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM配置"""
    provider: Literal['openai', 'dashscope', 'ollama'] = 'ollama'
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


class Neo4jConfig(BaseModel):
    """Neo4j知识图谱数据库连接配置"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0
