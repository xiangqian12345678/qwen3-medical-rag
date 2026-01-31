"""网络搜索模块"""
import sys
from pathlib import Path

# 添加当前项目目录到 Python 路径（支持直接运行）
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# 尝试相对导入（当作为包导入时）
from .web_search import create_web_search_tool

__all__ = [
    'create_web_search_tool'
]
