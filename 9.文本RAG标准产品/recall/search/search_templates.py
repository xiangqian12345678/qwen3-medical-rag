"""搜索相关Prompt模板"""
from typing import Dict, Union


# =============================================================================
# 检索router模板
# =============================================================================
WEB_SEARCH_JUDGE_SYSTEM_PROMPT = """你是一个智能助手。
请根据输入信息，判断所需信息是否完整，是否需要进行网络搜索。
如果没有必要，则need_search为false；
如果有必要，则思考缺失信息，输出need_search为true，
给出便于网络检索的`search_query`，并输出需保留文档的索引（1~N）。
数据格式

{format_instructions}"""

WEB_SEARCH_JUDGE_USER_PROMPT = """# 用户查询
{query}

# 检索到的文档
{docs}

# 输出示例
{{'need_search': true, 'search_query': '阿司匹林的副作用', 'remain_doc_index': [1, 3, 4]}}
{{'need_search': false, 'search_query': '', 'remain_doc_index': []}}"""


# =============================================================================
# 网络检索工具调用模板
# =============================================================================
CALLING_WEB_SEARCH_SYSTEM_PROMPT = """你是一个可以调用工具的智能助手。
你的任务是：当接收到用户查询时，必须使用 web_search 工具进行网络搜索，以获取最新的相关信息。

注意事项：
1. 对于任何查询，都应该使用 web_search 工具
2. web_search 工具的参数名为 query，值应该是优化后的搜索关键词
3. 不要直接回答问题，而是先调用工具获取信息
4. 工具名称为：web_search"""

CALLING_WEB_SEARCH_USER_PROMPT = """用户查询：{search_query}

请使用 web_search 工具对上述查询进行网络搜索。"""


# =============================================================================
# 模板注册表
# =============================================================================
PROMPT_TEMPLATES = {
    "web_router": {  # 网络搜索路由模板: 判断是否需要网络搜索
        "system": WEB_SEARCH_JUDGE_SYSTEM_PROMPT,
        "user": WEB_SEARCH_JUDGE_USER_PROMPT
    },

    "call_web": {  # 网络搜索调用模板: 执行网络搜索操作
        "system": CALLING_WEB_SEARCH_SYSTEM_PROMPT,
        "user": CALLING_WEB_SEARCH_USER_PROMPT
    },
}


def get_prompt_template(template_name: str) -> Dict[str, str]:
    """获取提示模板"""
    return PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["web_router"])


def register_prompt_template(name: str, template: Union[Dict[str, str], str]):
    """注册新的提示模板"""
    PROMPT_TEMPLATES[name] = template
