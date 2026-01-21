"""知识图谱检索相关Prompt模板"""
from typing import Dict, Union


# =============================================================================
# 知识图谱检索工具调用模板
# =============================================================================
CALLING_KGRAPH_SYSTEM_PROMPT = """你是一个可以调用工具的智能体，请根据输入的搜索查询调用知识图谱检索工具。"""

CALLING_KGRAPH_USER_PROMPT = """请对以下查询进行知识图谱检索：{query}"""


# =============================================================================
# 模板注册表
# =============================================================================
PROMPT_TEMPLATES = {
    "call_kgraph": {  # 知识图谱检索调用模板: 执行知识图谱检索操作
        "system": CALLING_KGRAPH_SYSTEM_PROMPT,
        "user": CALLING_KGRAPH_USER_PROMPT
    },
}


def get_prompt_template(template_name: str) -> Dict[str, str]:
    """获取提示模板"""
    return PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["call_kgraph"])


def register_prompt_template(name: str, template: Union[Dict[str, str], str]):
    """注册新的提示模板"""
    PROMPT_TEMPLATES[name] = template
