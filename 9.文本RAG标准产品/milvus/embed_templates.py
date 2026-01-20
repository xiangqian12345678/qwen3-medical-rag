"""向量数据库检索相关Prompt模板"""
from typing import Dict, Union


# =============================================================================
# 调用向量数据库查询模板
# =============================================================================
CALLING_DB_SYSTEM_PROMPT = """你是一个智能体，请根据输入查询和上下文，使用独立、自洽、便于进行单次且明确的向量检索查询文本，选择合适的检索参数，调用向量数据库进行查询。"""

CALLING_DB_USER_PROMPT = """用户查询：{query}"""


# =============================================================================
# 模板注册表
# =============================================================================
PROMPT_TEMPLATES = {
    "call_db": {  # 向量数据库查询模板: 调用向量库检索相关文档
        "system": CALLING_DB_SYSTEM_PROMPT,
        "user": CALLING_DB_USER_PROMPT
    },
}


def get_prompt_template(template_name: str) -> Dict[str, str]:
    """获取提示模板"""
    return PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["call_db"])


def register_prompt_template(name: str, template: Union[Dict[str, str], str]):
    """注册新的提示模板"""
    PROMPT_TEMPLATES[name] = template
