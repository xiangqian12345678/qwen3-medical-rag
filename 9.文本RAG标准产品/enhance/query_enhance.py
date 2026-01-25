"""多轮对话Agent：多轮医疗对话 + 规划式 RAG Agent"""
import logging
from typing import List

from langchain.output_parsers import OutputFixingParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage, AIMessage
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from enhance.agent_state import AgentState
from enhance_templates import get_prompt_template
from enhance.utils import strip_think_get_tokens

logger = logging.getLogger(__name__)


# 查询增强
# 1.意图识别 这部分还没有想好
class Intent(BaseModel):
    """LLM 用于判断是否需要拆分多个子查询的结构"""
    intent: str = Field(
        default="",
        description="意图"
    )


def intent_recognition(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
    根据业务补充，这部分需要大量业务抽象，主要针对milvus
    1.类别预测
    2.实体识别
    3.字段过滤
    """
    pass


# 2.query改写
class RewriteQuery(BaseModel):
    query: str = Field(
        default="",
        description="对原始问题的检索友好改写"
    )


def query_rewrite(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
    判断是否需要改写query，并格局需要优化query
    """
    # ========================================================
    # 步骤 1: 创建解析器（将 LLM 输出解析为 SplitQuery 结构化对象）
    # ========================================================
    parser = PydanticOutputParser(pydantic_object=RewriteQuery)
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
            get_prompt_template("rewrite_query")["system"].format(
                format_instructions=parser.get_format_instructions()
                .replace("{", "{{")
                .replace("}", "}}"),
                summary=multi_summary_text,
            ),
        ),
        # 插入主对话历史（不含追问过程的完整对话）
        MessagesPlaceholder(variable_name="dialogue_messages"),
        # 用户消息：背景信息和当前问题
        ("user", get_prompt_template("rewrite_query")["user"]),
    ])

    # ========================================================
    # 步骤 4: 构建执行链并调用 LLM
    # ========================================================
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "background_info": state["background_info"],
        "question": state["asking_messages"][-1][0].content,
        "dialogue_messages": state["dialogue_messages"],
    })

    # ========================================================
    # 步骤 5: 解析 LLM 输出
    # ========================================================
    # 处理 LLM 返回空结果或解析失败的情况
    if not ai["msg"] or not ai["msg"].strip():
        logger.warning(f"LLM 返回空结果，使用默认值:  rewrite_query=原始问题")
        rewrite_query = RewriteQuery(query=state["asking_messages"][-1][0].content)
    else:
        try:
            # 解析原理同 ask_judge 节点：
            # 1) Prompt 中的 format_instructions 提供 RewriteQuery 的 JSON schema
            # 2) LLM 根据 schema 生成结构化 JSON（如: {"need_rewrite": true, "rewrite_query": "..."}）
            # 3) OutputFixingParser 容错解析，自动修复格式错误
            # 4) 最终得到 rewriteQuery = RewriteQuery(need_rewrite=True, rewrite_query="...")
            rewrite_query: RewriteQuery = fixing.parse(ai["msg"])
        except Exception as e:
            logger.error(f"解析 LLM 输出失败: {e}, 输出内容: {ai['msg'][:200]}")
            # 解析失败时使用默认值
            rewrite_query = RewriteQuery(query=state["asking_messages"][-1][0].content)

    # ========================================================
    # 步骤 6: 更新状态
    # ========================================================
    # 保存查询改写结果
    # 示例结果: state["sub_query"] = RewriteQuery(need_rewrite=True, rewrite_query="头痛伴随低烧的诊断标准")
    state["rewrite_query"] = rewrite_query
    state["performance"].append(("rewrite_query", ai))
    return state


# 3.完善问题
class AskMess(BaseModel):
    """LLM 用于判断是否需要继续向用户追问的信息结构"""
    need_ask: bool = Field(
        default=False,
        description="根据已有信息，是否需要继续向用户追问"
    )
    questions: List[str] = Field(
        default_factory=list,
        description="需要向用户询问的问题列表"
    )


def query_refine(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
        判断是否需要向用户继续追问关键信息

        功能说明：
        1. 根据用户输入和已有背景信息，使用 LLM 判断是否需要追问
        2. 如果需要追问，生成追问问题列表
        3. 将追问对话记录到 asking_messages
        4. 增加 curr_ask_num 计数器
    """
    # ========================================================
    # 步骤 1: 创建解析器（将 LLM 输出解析为结构化对象）
    # ========================================================
    parser = PydanticOutputParser(pydantic_object=AskMess)  # 解析器: 将LLM输出解析为AskMess结构化对象
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)  # 修复解析器: 当LLM输出格式错误时自动修复

    # ========================================================
    # 步骤 2: 构建对话提示模板
    # ========================================================
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",  # 系统消息: 定义AI的角色和行为规范
            get_prompt_template("ask_user")["system"].format(
                # 获取Pydantic模型的JSON格式说明,并转义大括号避免与Python的.format()冲突
                format_instructions=parser.get_format_instructions()
                .replace("{", "{{")  # 将{转义为{{,避免被.format()当作变量占位符
                .replace("}", "}}")  # 将}转义为}},避免被.format()当作变量占位符
            ),
        ),
        MessagesPlaceholder(variable_name="asking_history"),  # 对话历史占位符: 动态插入追问历史消息
        ("human", get_prompt_template("ask_user")["user"]),  # 用户消息: 当前的用户提问和背景信息
    ])

    # ========================================================
    # 步骤 3: 获取当前追问轮次的历史消息
    # ========================================================
    # curr_ask_num=0 表示首轮追问，取空列表；否则取最后一轮追问历史
    curr_ask_history = [] if state["curr_ask_num"] == 0 else state["asking_messages"][-1]

    # ========================================================
    # 步骤 4: 构建执行链并调用 LLM
    # ========================================================
    # 执行链示例:
    #   prompt（模板）→ 填充变量 → llm（生成）→ strip_think_get_tokens（清理）→ 返回字典
    #
    # 实际执行示例:
    #   输入: {"background_info": "患者男,35岁", "question": "我头痛", "asking_history": []}
    #   ↓ prompt 格式化后发送给 LLM
    #   ↓ LLM 返回 AIMessage(content='{\n  "need_ask": true,\n  "questions": ["头痛持续多久了?", "有发热吗?"]\n}')
    #   ↓ strip_think_get_tokens 处理
    #   ai = {
    #     "msg": '{\n  "need_ask": true,\n  "questions": ["头痛持续多久了?", "有发热吗?"]\n}',
    #     "msg_len": 70,
    #     "msg_token_len": 35,
    #     "generate_time": 0.8
    #   }
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "background_info": state["background_info"],
        "question": state["curr_input"],
        "asking_history": curr_ask_history,
    })

    # ========================================================
    # 步骤 5: 将当前用户输入记录到对话历史
    # ========================================================
    if state["curr_ask_num"] == 0:
        # 首轮追问：创建新的对话组，添加用户消息
        state["asking_messages"].append([HumanMessage(content=state["curr_input"])])
    else:
        # 后续追问：追加到最后一组对话
        state["asking_messages"][-1].append(HumanMessage(content=state["curr_input"]))

    # ========================================================
    # 步骤 6: 解析 LLM 输出
    # ========================================================
    # 处理 LLM 返回空结果的情况
    if not ai["msg"] or not ai["msg"].strip():
        logger.warning(f"LLM 返回空结果，使用默认值: need_ask=False, questions=[]")
        patch = AskMess(need_ask=False, questions=[])
    else:
        try:
            # 为何能解析：
            # 1) Prompt 中通过 format_instructions 告诉 LLM 输出符合 AskMess 结构的 JSON schema
            #    format_instructions 示例:
            #    {"properties": {"need_ask": {"title": "Need Ask", "type": "boolean"},
            #                   "questions": {"title": "Questions", "type": "array", "items": {"type": "string"}}},
            #     "required": ["need_ask"], "title": "AskMess", "type": "object"}
            # 2) LLM 根据 schema 生成符合格式的 JSON（如: {"need_ask": true, "questions": ["..."]}）
            # 3) PydanticOutputParser 将 JSON 文本解析为 AskMess 对象，自动进行类型验证和转换
            # 4) OutputFixingParser 提供容错机制，如果格式轻微错误会自动调用 LLM 修复
            # 5) 最终得到 patch = AskMess(need_ask=True, questions=["头痛持续多久了?", "有发热吗?"])
            patch: AskMess = fixing.parse(ai["msg"])
        except Exception as e:
            logger.error(f"解析 LLM 输出失败: {e}, 输出内容: {ai['msg'][:200]}")
            # 解析失败时使用默认值
            patch = AskMess(need_ask=False, questions=[])

    # ========================================================
    # 步骤 7: 更新状态并记录 AI 回复
    # ========================================================
    state["ask_obj"] = patch  # 保存追问判断结果

    # 根据 need_ask 决定追问内容
    if patch.need_ask:
        # 需要追问：将生成的问题列表作为 AI 回复
        state["asking_messages"][-1].append(
            AIMessage(content="\n".join(patch.questions))
        )
    else:
        # 不需要追问：记录固定回复
        state["asking_messages"][-1].append(
            AIMessage(content="不需要询问任何其他信息")
        )

    # 记录性能数据和增加追问计数
    state["performance"].append(("ask", ai))
    state["curr_ask_num"] += 1
    return state


# 4.生成摘要
def generate_summary(state: AgentState, llm: BaseChatModel) -> AgentState:
    """
       从多轮追问中抽取关键背景信息
       功能说明：
       1. 从最后一轮追问对话中提取患者关键背景信息
       2. 将抽取的背景信息写入 state["background_info"]
       3. 记录性能日志到 state["performance"]
    """
    # ========================================================
    # 步骤 1: 构建信息抽取的 Prompt 模板
    # ========================================================
    prompt = ChatPromptTemplate.from_messages([
        # 系统消息：定义抽取任务（从对话中提取事实性信息，严禁推测）
        ("system", get_prompt_template("extract_user_info")["system"]),
        # 对话历史占位符：插入最后一轮追问的完整对话记录
        MessagesPlaceholder(variable_name="asking_history"),
        # 用户消息：传递原始问题，作为抽取参考
        ("human", get_prompt_template("extract_user_info")["user"]),
    ])

    # ========================================================
    # 步骤 2: 构建执行链并调用 LLM
    # ========================================================
    # 执行链示例:
    #   prompt（模板）→ 填充变量 → llm（生成）→ strip_think_get_tokens（清理）→ 返回字典
    #
    # 实际执行示例:
    #   输入数据:
    #     question = "我头痛"
    #     asking_history = [
    #       HumanMessage(content="我头痛"),
    #       AIMessage(content="头痛持续多久了? 有发热吗?"),
    #       HumanMessage(content="三天了，有点低烧"),
    #       AIMessage(content="不需要询问任何其他信息")
    #     ]
    #
    #   ↓ prompt 格式化后发送给 LLM
    #   系统提示: "任务：基于对话历史和用户当前提问，严格抽取已经明确提到的事实性信息..."
    #   对话历史: [Human:我头痛, AI:头痛持续多久了?, Human:三天了，有点低烧, AI:不需要询问]
    #   用户消息: "用户当前提问: 我头痛"
    #
    #   ↓ LLM 返回 AIMessage
    #   AIMessage(content='患者主诉头痛，持续3天，伴有低烧症状')
    #
    #   ↓ strip_think_get_tokens 处理
    #   ai = {
    #     "msg": "患者主诉头痛，持续3天，伴有低烧症状",
    #     "msg_len": 20,
    #     "msg_token_len": 15,
    #     "generate_time": 0.5
    #   }
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        # 获取最后一轮追问的原始问题（asking_messages[-1][0] 是该轮首次用户输入）
        "question": state["asking_messages"][-1][0].content,
        # 获取最后一轮追问的完整对话历史（包含用户和AI的交互）
        "asking_history": state["asking_messages"][-1],
    })

    # ========================================================
    # 步骤 3: 更新状态
    # ========================================================
    # 将 LLM 抽取的背景信息保存到 state
    # 示例结果: state["background_info"] = "患者主诉头痛，持续3天，伴有低烧症状"
    state["background_info"] = ai["msg"]
    # 记录性能数据（用于调试和监控）
    state["performance"].append(("extract", ai))
    return state
