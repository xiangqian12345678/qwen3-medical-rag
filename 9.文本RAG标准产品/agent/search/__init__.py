"""网络搜索模块"""
import sys
from pathlib import Path

# 添加当前项目目录到 Python 路径（支持直接运行）
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

try:
    # 尝试相对导入（当作为包导入时）
    from .web_search import llm_network_search, create_web_search_tool
except ImportError:
    # 回退到直接导入（当直接运行文件时）
    from search.web_search import llm_network_search, create_web_search_tool

# NetworkSearchResult 需要从 web_search 模块单独导入
try:
    from .web_search import NetworkSearchResult
except ImportError:
    from search.web_search import NetworkSearchResult

__all__ = ['llm_network_search', 'create_web_search_tool', 'NetworkSearchResult']
