"""知识图谱检索模块"""
import sys
from pathlib import Path

# 添加当前项目目录到 Python 路径（支持直接运行）
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

try:
    # 尝试相对导入（当作为包导入时）
    from .kgraph_search import llm_kgraph_search, create_kgraph_search_tool
    from .kg_templates import PROMPT_TEMPLATES as KG_PROMPT_TEMPLATES
    from .kg_templates import get_prompt_template as get_kg_prompt_template
    from .kg_templates import register_prompt_template as register_kg_prompt_template
except ImportError:
    # 回退到直接导入（当直接运行文件时）
    from kgraph.kgraph_search import llm_kgraph_search, create_kgraph_search_tool
    from kgraph.kg_templates import PROMPT_TEMPLATES as KG_PROMPT_TEMPLATES
    from kgraph.kg_templates import get_prompt_template as get_kg_prompt_template
    from kgraph.kg_templates import register_prompt_template as register_kg_prompt_template

__all__ = [
    'llm_kgraph_search', 'create_kgraph_search_tool',
    'KG_PROMPT_TEMPLATES', 'get_kg_prompt_template', 'register_kg_prompt_template'
]
