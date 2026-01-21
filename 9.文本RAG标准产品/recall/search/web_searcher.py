"""数据库工厂类，用于创建和管理知识库实例"""
import json
import logging
from typing import Dict, Any, Optional, Union

from duckduckgo_search import DDGS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 知识库实例缓存
_ws_instance: Optional["WebSearcher"] = None


class WebSearcher:
    """知识库类，封装向量数据库检索功能"""

    def __init__(self, config: Union[Dict[str, Any], None] = None):
        pass

    def search(self, query: str, cnt: int = 5) -> list[Document]:
        """真正去拉取网页结果的搜索函数"""
        docs = []
        with DDGS() as ddgs:
            for i, hit in enumerate(
                    ddgs.text(query, max_results=cnt), start=1
            ):
                docs.append(
                    Document(
                        page_content=hit["body"],
                        metadata={
                            "title": hit["title"],
                            "url": hit["href"],
                            "rank": i,
                        },
                    )
                )
        return docs


def get_ws(config: Dict[str, Any] = None) -> WebSearcher:
    """
    获取知识库实例（单例模式）

    Args:
        config: 配置字典

    Returns:
        知识库实例
    """
    global _ws_instance

    if _ws_instance is None:
        _ws_instance = WebSearcher(config)

    return _ws_instance


def reset_kb():
    """重置知识库实例（用于测试或重新初始化）"""
    global _ws_instance
    _ws_instance = None
