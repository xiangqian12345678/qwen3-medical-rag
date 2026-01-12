"""Prompt模板"""
from typing import Dict, Union


# =============================================================================
# 基础RAG提示模板
# =============================================================================
BASIC_RAG_SYSTEM_PROMPT = """你是一名专业的医学知识助手，能够基于提供的医学资料准确回答用户问题。"""

BASIC_RAG_USER_PROMPT = """# 要求
1. 必须严格基于提供的参考资料回答问题
2. 如果参考资料中没有相关信息，请明确说明"根据提供的资料无法回答此问题"
3. 回答要专业、准确，同时通俗易懂，不需要长篇大论
4. 不要编造或推测未在资料中提及的信息
5. 如涉及具体诊疗建议，请提醒用户咨询专业医生

# 参考资料
{all_document_str}

# 用户问题
{input}

请基于以上参考资料回答用户问题。如果资料不足以回答问题，请如实说明。"""


# =============================================================================
# 模板注册表
# =============================================================================
PROMPT_TEMPLATES = {
    "basic_rag": {
        "system": BASIC_RAG_SYSTEM_PROMPT,
        "user": BASIC_RAG_USER_PROMPT
    },
}


def get_prompt_template(template_name: str) -> Union[Dict[str, str], str]:
    """
    获取提示模板

    Args:
        template_name: 模板名称

    Returns:
        模板内容或默认模板
    """
    return PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["basic_rag"])
