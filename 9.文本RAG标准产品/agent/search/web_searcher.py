"""数据库工厂类，用于创建和管理知识库实例"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from duckduckgo_search import DDGS
from langchain_core.documents import Document

# 添加当前项目目录到 Python 路径（支持直接运行）
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# 导入配置和工具函数
try:
    # 尝试相对导入（当作为包导入时）
    from ..config.models import AppConfig, SearchRequest
    from ..agent.utils import create_embedding_client
except ImportError:
    # 回退到绝对导入（当直接运行文件时）
    from config.models import AppConfig, SearchRequest
    from agent.utils import create_embedding_client

logger = logging.getLogger(__name__)

# 知识库实例缓存
_ws_instance: Optional["WebSearcher"] = None


class WebSearcher:
    """知识库类，封装向量数据库检索功能"""

    def __init__(self, config: AppConfig):
        pass

    def search(query: str, cnt: int = 5) -> list[Document]:
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
        if config is None:
            raise ValueError("首次调用必须传入config参数")

        from config.models import AppConfig
        app_config = AppConfig(**config) if isinstance(config, dict) else config
        _ws_instance = WebSearcher(app_config)

    return _ws_instance


def reset_kb():
    """重置知识库实例（用于测试或重新初始化）"""
    global _ws_instance
    _ws_instance = None
