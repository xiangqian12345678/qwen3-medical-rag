"""Milvus向量检索模块"""
import sys
from pathlib import Path

# 添加当前项目目录到 Python 路径（支持直接运行）
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

try:
    # 尝试相对导入（当作为包导入时）
    from .embed_search import llm_db_search, create_db_search_tool
    from .embed_templates import PROMPT_TEMPLATES as EMBED_PROMPT_TEMPLATES
    from .embed_templates import get_prompt_template as get_embed_prompt_template
    from .embed_templates import register_prompt_template as register_embed_prompt_template
except ImportError:
    # 回退到直接导入（当直接运行文件时）
    from recall.milvus.embed_search import llm_db_search, create_db_search_tool
    from recall.milvus.embed_templates import PROMPT_TEMPLATES as EMBED_PROMPT_TEMPLATES
    from recall.milvus.embed_templates import get_prompt_template as get_embed_prompt_template
    from recall.milvus.embed_templates import register_prompt_template as register_embed_prompt_template

__all__ = [
    'llm_db_search', 'create_db_search_tool',
    'EMBED_PROMPT_TEMPLATES', 'get_embed_prompt_template', 'register_embed_prompt_template'
]
