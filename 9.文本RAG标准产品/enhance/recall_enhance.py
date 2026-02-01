"""多轮对话Agent：多轮医疗对话 + 规划式 RAG Agent"""
import logging

from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .agent_state import AgentState, MultiQueries, SubQueries, SuperordinateQuery, HypotheticalAnswer
from .enhance_templates import get_prompt_template
from .utils import invoke_with_timing

logger = logging.getLogger(__name__)


# 1.生成多个问题
def generate_multi_queries(state: AgentState, llm: BaseChatModel) -> AgentState:
    # 创建解析器（将 LLM 输出解析为 MultiQueries 结构化对象）
    parser = PydanticOutputParser(pydantic_object=MultiQueries)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # 构建多轮对话摘要
    multi_summary_text = "\n".join(state.get("multi_summary", [])) if state.get("multi_summary") else ""

    # 构建对话提示模板
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            get_prompt_template("multi_query")["system"].format(
                format_instructions=parser.get_format_instructions()
                .replace("{", "{{")
                .replace("}", "}}"),
                summary=multi_summary_text,
            ),
        ),
        # 插入主对话历史（不含追问过程的完整对话）
        MessagesPlaceholder(variable_name="dialogue_messages"),
        # 用户消息：背景信息和当前问题
        ("user", get_prompt_template("multi_query")["user"]),
    ])


    #  构建执行链并调用 LLM
    ai = invoke_with_timing(
        (prompt | llm),
        {
            "background_info": state["background_info"],
            "question": state["curr_input"],
            "dialogue_messages": state["dialogue_messages"],
        },
        stage_name="multi_query",
        state=state
    )

    #  解析 LLM 输出
    if not ai["msg"] or not ai["msg"].strip():
        logger.warning(f"LLM 返回空结果，使用默认值: need_split=False, sub_query=[], rewrite_query=原始问题")
        multi_queries = MultiQueries(queries=[])
    else:
        try:
            multi_queries: MultiQueries = fixing.parse(ai["msg"])
        except Exception as e:
            logger.error(f"解析 LLM 输出失败: {e}, 输出内容: {ai['msg'][:200]}")
            multi_queries = MultiQueries(queries=[])

    # 保存查询规划结果
    new_state = state.copy()
    new_state["multi_query"] = multi_queries
    new_state["performance"] = state["performance"] + [("multi_query", ai)]
    return new_state


# 2.生成子问题
def generate_sub_queries(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
        判断是否需要拆分多个子查询进行检索

        功能说明：
        1. 根据用户问题和背景信息，判断是否需要将问题拆分为多个子查询
        2. 如果需要拆分，生成最多 3 个相互独立的子查询
        3. 如果不需要拆分，生成一个检索友好的改写查询
        4. 将结果写入 state["sub_query"]
    """
    # ========================================================
    # 步骤 1: 创建解析器（将 LLM 输出解析为 SplitQuery 结构化对象）
    # ========================================================
    parser = PydanticOutputParser(pydantic_object=SubQueries)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # ========================================================
    # 步骤 2: 构建多轮对话摘要
    # ========================================================
    # 将历史对话摘要列表拼接为字符串，用于上下文理解
    # 示例: multi_summary = ["用户询问头痛问题", "用户提到持续三天且有低烧"]
    #       multi_summary_text = "用户询问头痛问题\n用户提到持续三天且有低烧"
    multi_summary_text = "\n".join(state.get("multi_summary", [])) if state.get("multi_summary") else ""

    # ========================================================
    # 步骤 3: 构建 Prompt 模板
    # ========================================================
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            # 格式化系统提示，包含:
            # 1) format_instructions: SplitQuery 的 JSON schema（转义大括号避免 .format() 冲突）
            # 2) summary: 历史对话摘要，帮助 LLM 理解上下文
            get_prompt_template("split_query")["system"].format(
                format_instructions=parser.get_format_instructions()
                .replace("{", "{{")
                .replace("}", "}}"),
                summary=multi_summary_text,
            ),
        ),
        # 插入主对话历史（不含追问过程的完整对话）
        MessagesPlaceholder(variable_name="dialogue_messages"),
        # 用户消息：背景信息和当前问题
        ("user", get_prompt_template("split_query")["user"]),
    ])

    # ========================================================
    # 步骤 4: 构建执行链并调用 LLM
    # ========================================================
    ai = invoke_with_timing(
        (prompt | llm),
        {
            "background_info": state["background_info"],
            "question": state["asking_messages"][-1][0].content,
            "dialogue_messages": state["dialogue_messages"],
        },
        stage_name="sub_query",
        state=state
    )

    # ========================================================
    # 步骤 5: 解析 LLM 输出
    # ========================================================
    # 处理 LLM 返回空结果或解析失败的情况
    if not ai["msg"] or not ai["msg"].strip():
        logger.warning(f"LLM 返回空结果，使用默认值: need_split=False, sub_query=[], rewrite_query=原始问题")
        sub_queries = SubQueries(need_split=False, queries=[])
    else:
        try:
            sub_queries: SubQueries = fixing.parse(ai["msg"])
        except Exception as e:
            logger.error(f"解析 LLM 输出失败: {e}, 输出内容: {ai['msg'][:200]}")
            # 解析失败时使用默认值
            sub_queries = SubQueries(need_split=False, queries=[])

    # ========================================================
    # 步骤 6: 限制子查询数量并更新状态
    # ========================================================
    sub_queries.queries = sub_queries.queries[:3]

    new_state = state.copy()
    new_state["sub_query"] = sub_queries
    new_state["performance"] = state["performance"] + [("sub_query", ai)]
    return new_state


# 3.上位问题优化
def generate_superordinate_query(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
        生成上位问题
    """
    # ========================================================
    # 步骤 1: 创建解析器（将 LLM 输出解析为 SuperordinateQuery 结构化对象）
    # ========================================================
    parser = PydanticOutputParser(pydantic_object=SuperordinateQuery)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # ========================================================
    # 步骤 2: 构建 Prompt 模板
    # ========================================================
    multi_summary_text = "\n".join(state.get("multi_summary", [])) if state.get("multi_summary") else ""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            # 格式化系统提示，包含:
            # 1) format_instructions: SuperordinateQuery 的 JSON schema（转义大括号避免 .format() 冲突）
            get_prompt_template("superordinate_query")["system"].format(
                format_instructions=parser.get_format_instructions()
                .replace("{", "{{")
                .replace("}", "}}"),
                summary=multi_summary_text,  # superordinate_query 不需要 summary 参数，传入空字符串
            ),
        ),
        # 插入主对话历史（不含追问过程的完整对话）
        MessagesPlaceholder(variable_name="dialogue_messages"),
        # 用户消息：背景信息和当前问题
        ("user", get_prompt_template("superordinate_query")["user"]),
    ])

    # ========================================================
    # 步骤 3: 构建执行链并调用 LLM
    # ========================================================
    ai = invoke_with_timing(
        (prompt | llm),
        {
            "background_info": state["background_info"],
            "question": state["curr_input"],
            "dialogue_messages": state["dialogue_messages"],
        },
        stage_name="superordinate_query",
        state=state
    )

    # ========================================================
    # 步骤 4: 解析 LLM 输出
    # ========================================================
    if not ai["msg"] or not ai["msg"].strip():
        logger.warning(f"LLM 返回空结果，使用默认值: superordinate_query=原始问题")
        superordinate_query = SuperordinateQuery(superordinate_query=state["curr_input"])
    else:
        try:
            superordinate_query: SuperordinateQuery = fixing.parse(ai["msg"])
        except Exception as e:
            logger.error(f"解析 LLM 输出失败: {e}, 输出内容: {ai['msg'][:200]}")
            superordinate_query = SuperordinateQuery(superordinate_query=state["curr_input"])

    # ========================================================
    # 步骤 5: 更新状态
    # ========================================================
    new_state = state.copy()
    new_state["superordinate_query"] = superordinate_query
    new_state["performance"] = state["performance"] + [("superordinate_query", ai)]
    return new_state


# 4.假设性答案
def generate_hypothetical_answer(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
        生成假设性答案
    """
    # ========================================================
    # 步骤 1: 创建解析器（将 LLM 输出解析为 HypotheticalAnswer 结构化对象）
    # ========================================================
    parser = PydanticOutputParser(pydantic_object=HypotheticalAnswer)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    # ========================================================
    # 步骤 2: 构建 Prompt 模板
    # ========================================================
    multi_summary_text = "\n".join(state.get("multi_summary", [])) if state.get("multi_summary") else ""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            # 格式化系统提示，包含:
            # 1) format_instructions: HypotheticalAnswer 的 JSON schema（转义大括号避免 .format() 冲突）
            get_prompt_template("hypothetical_answer")["system"].format(
                format_instructions=parser.get_format_instructions()
                .replace("{", "{{")
                .replace("}", "}}"),
                summary=multi_summary_text,  # hypothetical_answer 不需要 summary 参数，传入空字符串
            ),
        ),
        # 插入主对话历史（不含追问过程的完整对话）
        MessagesPlaceholder(variable_name="dialogue_messages"),
        # 用户消息：背景信息和当前问题
        ("user", get_prompt_template("hypothetical_answer")["user"]),
    ])

    # 步骤 3: 构建执行链并调用 LLM
    ai = invoke_with_timing(
        (prompt | llm),
        {
            "background_info": state["background_info"],
            "question": state["curr_input"],
            "dialogue_messages": state["dialogue_messages"],
        },
        stage_name="hypothetical_answer",
        state=state
    )

    # 步骤 4: 解析 LLM 输出
    if not ai["msg"] or not ai["msg"].strip():
        logger.warning(f"LLM 返回空结果，使用默认值: hypothetical_answer=原始问题")
        hypothetical_answer = HypotheticalAnswer(hypothetical_answer=state["curr_input"])
    else:
        try:
            hypothetical_answer: HypotheticalAnswer = fixing.parse(ai["msg"])
        except Exception as e:
            logger.error(f"解析 LLM 输出失败: {e}, 输出内容: {ai['msg'][:200]}")
            hypothetical_answer = HypotheticalAnswer(hypothetical_answer=state["curr_input"])

    # 步骤 5: 更新状态
    new_state = state.copy()
    new_state["hypothetical_answer"] = hypothetical_answer
    new_state["performance"] = state["performance"] + [("hypothetical_answer", ai)]
    return new_state
