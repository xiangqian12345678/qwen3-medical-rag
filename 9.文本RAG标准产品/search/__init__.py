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
    from .web_search import NetworkSearchResult
    from .search_templates import PROMPT_TEMPLATES as SEARCH_PROMPT_TEMPLATES
    from .search_templates import get_prompt_template as get_search_prompt_template
    from .search_templates import register_prompt_template as register_search_prompt_template
except ImportError:
    # 回退到直接导入（当直接运行文件时）
    from search.web_search import llm_network_search, create_web_search_tool
    from search.web_search import NetworkSearchResult
    from search.search_templates import PROMPT_TEMPLATES as SEARCH_PROMPT_TEMPLATES
    from search.search_templates import get_prompt_template as get_search_prompt_template
    from search.search_templates import register_prompt_template as register_search_prompt_template

__all__ = [
    'llm_network_search', 'create_web_search_tool', 'NetworkSearchResult',
    'SEARCH_PROMPT_TEMPLATES', 'get_search_prompt_template', 'register_search_prompt_template'
]
