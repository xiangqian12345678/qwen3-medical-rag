"""RAG相关Prompt模板"""
from typing import Dict, Union

from langchain_core.prompts import PromptTemplate

# =============================================================================
# 单轮主动问讯模板
# =============================================================================
ASK_SYSTEM_PROMPT = """
你是一名资深的医务人员。
任务：根据当前已知的基本背景信息、对话历史以及当前用户提问，
判断是否需要向用户追加追问来弥补关键信息缺口，并产出一个小而精的追问清单（1~3个问题）。
输出必须严格遵循结构：need_ask(bool), questions(list[str])

# 判断标准
1) 综合考虑对话历史中用户已经提供的所有信息
2) 如果用户在当前输入中回答了之前追问的问题，应该识别并记录这些信息
3) 只有当关键信息确实缺失且无法从已有信息中推断时，才继续追问
4) 如果用户已经提供了足够的信息来生成有意义的回答，则设置 need_ask=false

# 问题设计规范
1) 每个问题只问一个事实点，语言简洁、口语化，避免专业术语与引导性表述；控制在15~30个汉字
2) 按影响决策的优先级排序（安全相关在前）
3) 问题应便于用户用简单短语或数字回答（如"开始于何时？"、"是否发热，最高到几度？"）
4) 请仅问讯一些关键性问题，反复问讯会影响用户体验
5) 不要重复追问用户已经回答过的问题（即使回答不完整，也要避免重复询问同一事实点）

# 禁止事项
1) 不提出诊断性或引导性问题（如"是不是阑尾炎？"），如果有必要的话，可以问讯用户的过往病史（如"是否当前患有或者曾经患有高血压？"）
2) 不一次抛出过多问题；若缺口很多，仅保留最关键的1~3个
3) 不输出除指定键之外的任何内容

# JSON格式
{format_instructions}
"""

ASK_USER_PROMPT = """
# 基本背景信息
{background_info}

# 对话历史
{asking_history}

# 用户当前输入
{question}
"""

# =============================================================================
# 总结用户信息模板
# =============================================================================
EXTRACT_USER_INFO_SYSTEM_PROMPT = """
任务：基于对话历史和用户当前提问，
严格抽取已经明确提到的事实性信息，生成简短的背景摘要。

# 输出要求
1) 只保留对话中出现过的具体信息。
2) 严禁添加、推测、联想或扩展任何未出现的信息。
3) 未提及的信息请不要补充、假设或者推断。
4) 输出风格应简洁、客观，避免使用"可能""需注意"等推测性表达。
"""

EXTRACT_USER_INFO_USER_PROMPT = """
# 用户当前提问
{question}"""

# =============================================================================
# 拆分为子查询模板
# =============================================================================
QUERY_SPLIT_SYSTEM_PROMPT = """
你是一名资深的医务人员。
任务：根据历史对话、摘要、以及所掌握的用户信息，结合用户当前问题，判断是否需要将查询拆解为多个独立的子查询。
输出必须严格遵循结构： need_split(bool), sub_query(list[str])  

# 查询输出规范
1) 如果用户问题口语化、模糊、或无法直接作为检索关键词使用，则需重写（need_split=True，填写rewrite_query）。
2) 不输出除指定键之外的任何内容。
3) 如果用户问题本身已经清晰且适合作为检索词，则need_split=False，rewrite_query原样输出。

# JSON格式
{format_instructions}

# 前文摘要
{summary}
"""

QUERY_SPLIT_USER_PROMPT = """
# 基本背景信息
{background_info}

# 用户当前提问
{question}
"""

# =============================================================================
# 改写查询模板
# =============================================================================
QUERY_REWRITE_SYSTEM_PROMPT = """
你是一名医学检索优化师。
任务：把用户口语化、模糊或缺医学实体的提问，改写成一句专业、可检索的短句。
输出必须严格遵循结构：need_rewrite(bool), sub_query(list[str]) 

1) 每一个查询都应该是一个独立的，意图清晰的，简短的，医学专业化的句子，便于向量检索。
2) 不输出除指定键之外的任何内容。
3) 如果用户问题口语化、模糊、或无法直接作为检索关键词使用，则需重写（need_split=True，填写rewrite_query）。
4) 如果用户的问题需要多步查询核实事实（例如需要分别确认症状、检查、治疗），则need_split=True，并在sub_query中填入子查询，rewrite_query留空。
5) 拆解查询和重写查询不会同时发生，因为重写是单次查询，拆解是多次查询。
6) 如果用户问题本身已经清晰且适合作为检索词，则need_split=False，rewrite_query原样输出。

# JSON格式
{format_instructions}

# 前文摘要
{summary}
"""

QUERY_REWRITE_USER_PROMPT = """ 
# 基本背景信息
{background_info}

# 用户当前提问
{question}
"""

# =============================================================================
# 评判模板
# =============================================================================
JUDGE_RAG_SYSTEM_PROMPT = """
根据文档内容和用户查询，你需要判断模型摘要是否遵循了事实，
模型摘要是否脱离了文档内容，可能存在编造或推测。仅输出Y或N 
"""

JUDGE_RAG_USER_PROMPT = """# 文档内容:
{format_document_str}

# 用户查询:
{query}

# 模型摘要:
{summary}

# 输出示例:
# N

# 输出:
"""

# =============================================================================
# 多样化查询模板
# =============================================================================
MULTI_QUERY_SYSTEM_PROMPT = """
你是一名医学检索优化师。
任务：
针对用户给出的问题，生成 5 个不同版本的改写问题，以便从向量数据库中检索相关文档。
通过从多个角度改写用户问题，帮助用户克服基于距离相似度搜索的局限。
请将这些替代问题用换行符分隔。
输出必须严格遵循结构：mult_query(list[str])  

# 查询输出规范
1) 每一个查询都应该是一个独立的，表达方式不同的，语义一致的，意图清晰的，简短的，医学专业化的句子，便于向量检索。
2) 不输出除指定键之外的任何内容。


# JSON格式
{format_instructions}

# 前文摘要
{summary}
"""

MULTI_QUERY_USER_PROMPT = """ 
# 基本背景信息
{background_info}

用户当前提问：{question}
"""

# =============================================================================
# 上位问题查询模板
# =============================================================================
step_back_prompt = PromptTemplate.from_template(
    """
    基于以下问题，生成一个更抽象的上位问题：
    原始问题: {original_question}
    上位问题:
    """
)

SUPERORDINATE_QUERY_SYSTEM_PROMPT = """
你是一个医学检索优化师。
任务：
针对用户给出的问题，生成一个更抽象的上位问题，以便从向量数据库中检索相关文档。
输出必须严格遵循结构：superordinate_query(str)  

# 查询输出规范
1) 不输出除指定键之外的任何内容。


# JSON格式
{format_instructions}

# 前文摘要
{summary}
"""

SUPERORDINATE_QUERY_USER_PROMPT = """ 
# 基本背景信息
{background_info}

用户当前提问：{question}
"""

# =============================================================================
# 上位问题查询模板
# =============================================================================
HYPOTHETICAL_ANSWER_SYSTEM_PROMPT = """
你是一个医学检索优化师。
任务：
针对用户给出的问题，生成一个假设回答，以便从向量数据库中检索相关文档。
输出必须严格遵循结构：hypothetical_answer(str)  

# 查询输出规范
1) 不输出除指定键之外的任何内容。

# JSON格式
{format_instructions}

# 前文摘要
{summary}
"""

HYPOTHETICAL_ANSWER_USER_PROMPT = """ 
# 基本背景信息
{background_info}

用户当前提问：{question}
"""

# =============================================================================
# 模板注册表
# =============================================================================
PROMPT_TEMPLATES = {
    "ask_user": {  # 主动追问模板: 判断是否需要向用户追问关键信息
        "system": ASK_SYSTEM_PROMPT,
        "user": ASK_USER_PROMPT
    },

    "extract_user_info": {  # 用户信息提取模板: 从对话中抽取背景信息
        "system": EXTRACT_USER_INFO_SYSTEM_PROMPT,
        "user": EXTRACT_USER_INFO_USER_PROMPT
    },

    "rewrite_query": {  # 查询处理模板: 重写用户查询
        "system": QUERY_REWRITE_SYSTEM_PROMPT,
        "user": QUERY_REWRITE_USER_PROMPT
    },

    "split_query": {  # 查询处理模板: 拆分用户查询
        "system": QUERY_SPLIT_SYSTEM_PROMPT,
        "user": QUERY_SPLIT_USER_PROMPT
    },

    "multi_query": {  # 查询处理模板： 生成多样化查询
        "system": MULTI_QUERY_SYSTEM_PROMPT,
        "user": MULTI_QUERY_USER_PROMPT
    },

    "superordinate_query": {  # 上位查询模板： 生成上位查询
        "system": SUPERORDINATE_QUERY_SYSTEM_PROMPT,
        "user": SUPERORDINATE_QUERY_USER_PROMPT
    },

    "hypothetical_answer":{ # 假设回答模板： 生成假设回答
        "system":HYPOTHETICAL_ANSWER_SYSTEM_PROMPT,
        "user":HYPOTHETICAL_ANSWER_USER_PROMPT
    }
}


def get_prompt_template(template_name: str) -> Dict[str, str]:
    """获取提示模板"""
    return PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["ask_user"])


def register_prompt_template(name: str, template: Union[Dict[str, str], str]):
    """注册新的提示模板"""
    PROMPT_TEMPLATES[name] = template
