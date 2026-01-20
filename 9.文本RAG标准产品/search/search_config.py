"""搜索配置数据模型"""
from pydantic import BaseModel, Field


class SearchAgentConfig(BaseModel):
    """搜索Agent配置"""
    network_search_enabled: bool = False
    network_search_cnt: int = Field(default=10, gt=0, description="网络搜索结果最大数")
