"""知识图谱配置数据模型"""
from pydantic import BaseModel


class Neo4jConfig(BaseModel):
    """Neo4j知识图谱数据库连接配置"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_recall_num: int = 10
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0
