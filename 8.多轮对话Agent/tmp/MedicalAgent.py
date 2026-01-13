"""
MedicalAgent：多轮医疗对话 + 规划式 RAG Agent
=================================================

整体职责：
1. 判断是否需要向用户追问关键信息（Ask）
2. 从多轮追问中抽取背景信息（Extract）
3. 判断是否需要拆分为多个子查询（Plan / Split）
4. 并行执行多个 SearchGraph（Parallel Execute）
5. 汇总子查询结果，写回对话（Answer / Synthesize）

该文件是【完整可运行代码 + 架构级中文注释】，用于：
- 代码审查
- 二次开发
- 架构讲解 / 交接
"""

# =========================
# 基础依赖导入
# =========================
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Tuple
from typing_extensions import TypedDict
from functools import partial
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# LangChain / LangGraph 相关
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from langgraph.constants import END

# 项目内依赖
from ..config.models import AppConfig
from ..core.utils import create_llm_client
from .SearchGraph import SearchMessagesState, SearchGraph
from MedicalRag.prompts.templates import get_prompt_template
from .utils import strip_think_get_tokens

logger = logging.getLogger(__name__)


# =========================================================
# 一、结构化模型定义（LLM 输出约束）
# =========================================================

class AskMess(BaseModel):
    """
    LLM 用于判断是否需要继续向用户追问的信息结构
    """
    need_ask: bool = Field(
        default=False,
        description="根据已有信息，是否需要继续向用户追问"
    )
    questions: List[str] = Field(
        default_factory=list,
        description="需要向用户询问的问题列表"
    )


class SplitQuery(BaseModel):
    """
    LLM 用于判断是否需要拆分多个子查询的结构
    """
    need_split: bool = Field(
        default=False,
        description="是否需要拆分为多个独立子查询"
    )
    sub_query: List[str] = Field(
        default_factory=list,
        description="子查询列表（最多 3 个，相互独立）"
    )
    rewrite_query: str = Field(
        default="",
        description="不拆分时，对原始问题的检索友好改写"
    )


# =========================================================
# 二、LangGraph 顶层 State 定义
# =========================================================

class MedicalAgentState(TypedDict, total=False):
    """
    MedicalAgent 在 LangGraph 中流转的【唯一状态对象】
    """

    # ---------- 对话与上下文 ----------
    dialogue_messages: List[BaseMessage]  # 主对话历史
    asking_messages: List[List[BaseMessage]]  # 每一轮追问形成一组子对话
    background_info: str  # 从追问中抽取的背景摘要
    curr_input: str  # 当前用户输入

    # ---------- 规划结果 ----------
    ask_obj: AskMess  # 是否需要继续追问
    sub_query: SplitQuery  # 子查询规划结果
    sub_query_results: List[SearchMessagesState]  # 并行子图执行结果

    # ---------- 控制变量 ----------
    max_ask_num: int  # 最大追问轮次
    curr_ask_num: int  # 当前已追问次数

    # ---------- 输出 & 调试 ----------
    final_answer: str  # 最终答案（可供 UI 使用）
    performance: List[Any]  # 调试 / 性能信息


# =========================================================
# 三、节点函数定义（LangGraph Nodes）
# =========================================================


def ask_judge(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """
    【节点 1】判断是否需要向用户继续追问关键信息

    功能说明：
    1. 根据用户输入和已有背景信息，使用 LLM 判断是否需要追问
    2. 如果需要追问，生成追问问题列表
    3. 将追问对话记录到 asking_messages
    4. 增加 curr_ask_num 计数器

    输入样例（首次提问）：
        state = {
            "curr_input": "鼻中隔手术后一直不舒服",
            "background_info": "",
            "curr_ask_num": 0,
            "asking_messages": [],
        }

    输出样例（需要追问）：
        state = {
            "curr_input": "鼻中隔手术后一直不舒服",
            "ask_obj": {"need_ask": True, "questions": ["手术多久了？", "是否有后鼻滴漏？"]},
            "asking_messages": [
                [
                    HumanMessage("鼻中隔手术后一直不舒服"),
                    AIMessage("手术多久了？\n是否有后鼻滴漏？"),
                ],
            ],
            "curr_ask_num": 1,
            "performance": [("ask", {...})],
        }

    输入样例（用户回复追问）：
        state = {
            "curr_input": "大概一周了",
            "background_info": "",
            "curr_ask_num": 1,
            "asking_messages": [
                [
                    HumanMessage("鼻中隔手术后一直不舒服"),
                    AIMessage("手术多久了？\n是否有后鼻滴漏？"),
                ],
            ],
        }

    输出样例（不需要追问）：
        state = {
            "curr_input": "大概一周了",
            "ask_obj": {"need_ask": False, "questions": []},
            "asking_messages": [
                [
                    HumanMessage("鼻中隔手术后一直不舒服"),
                    AIMessage("手术多久了？\n是否有后鼻滴漏？"),
                    HumanMessage("大概一周了"),
                    AIMessage("不需要询问任何其他信息"),
                ],
            ],
            "curr_ask_num": 2,
            "performance": [("ask", {...})],
        }
    """

    # Pydantic 输出解析器 + 自动修复
    parser = PydanticOutputParser(pydantic_object=AskMess)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    '''
    Prompt 结构：System + 历史追问 + User
    最终传递给 LLM 的示例:
        System:
        你是一名资深的医务人员。任务：根据当前已知的基本背景信息结合当前用户提问...
        输出必须严格遵循结构：need_ask(bool), questions(list[str])
        # JSON格式
        {{"description": "...", "type": "object", ...}}  ← 双花括号保持原样
        
        History:
        Human: 鼻中隔手术后一直不舒服
        AI: 手术多久了？是否有后鼻滴漏？
        
        User:
        # 基本背景信息
        无
        # 用户当前输入
        大概一周了
    '''

    prompt = ChatPromptTemplate.from_messages([
        # # 第一层：System 角色定义 + 输出格式约束
        (
            "system",
            get_prompt_template("ask_user")["system"].format(
                format_instructions=parser.get_format_instructions()
                .replace("{", "{{")
                .replace("}", "}}")
            ),
        ),
        # 第二层：历史追问记录（可变长度）
        MessagesPlaceholder(variable_name="asking_history"),
        # 第三层：当前用户输入
        ("human", get_prompt_template("ask_user")["user"]),
    ])

    # 当前追问上下文
    curr_ask_history = [] if state["curr_ask_num"] == 0 else state["asking_messages"][-1]

    # 调用 LLM
    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "background_info": state["background_info"],
        "question": state["curr_input"],
        "asking_history": curr_ask_history,
    })

    # 初始化 / 追加追问对话
    if state["curr_ask_num"] == 0:
        state["asking_messages"].append([HumanMessage(content=state["curr_input"])])
    else:
        state["asking_messages"][-1].append(HumanMessage(content=state["curr_input"]))

    # 解析 LLM 输出
    patch: AskMess = fixing.parse(ai["msg"])
    state["ask_obj"] = patch

    # 写入 AI 回复
    if patch.need_ask:
        state["asking_messages"][-1].append(
            AIMessage(content="\n".join(patch.questions))
        )
    else:
        state["asking_messages"][-1].append(
            AIMessage(content="不需要询问任何其他信息")
        )

    state["performance"].append(("ask", ai))
    state["curr_ask_num"] += 1
    return state


def route_ask_again(state: MedicalAgentState) -> str:
    """
    LangGraph 条件路由函数

    功能说明：
    1. 判断是否需要继续向用户追问
    2. 若仍需追问且未超过最大次数 → 返回 "ask"（流程到 END，等待用户回复）
    3. 否则 → 返回 "pass"（进入 extract 节点，开始信息抽取）

    输入样例（需要继续追问）：
        state = {
            "ask_obj": {"need_ask": True, "questions": [...]},
            "curr_ask_num": 1,
            "max_ask_num": 5,
        }

    输出样例：
        "ask"

    输入样例（不需要追问或达到最大次数）：
        state = {
            "ask_obj": {"need_ask": False, "questions": []},
            "curr_ask_num": 2,
            "max_ask_num": 5,
        }

    输出样例：
        "pass"
    """
    return (
        "ask"
        if state["ask_obj"].need_ask and state["curr_ask_num"] < state["max_ask_num"]
        else "pass"
    )


def extract_background_info(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """
    【节点 2】从多轮追问中抽取关键背景信息

    功能说明：
    1. 从最后一轮追问对话中提取患者关键背景信息
    2. 将抽取的背景信息写入 state["background_info"]
    3. 记录性能日志到 state["performance"]

    输入样例：
        state = {
            "asking_messages": [
                [
                    HumanMessage("我最近做了鼻中隔手术"),
                    AIMessage("请问手术多久了？是否有后鼻滴漏？"),
                    HumanMessage("三年前做的手术，一直有后鼻滴漏"),
                    AIMessage("不需要询问任何其他信息"),
                ],
            ],
        }

    输出样例：
        state = {
            "asking_messages": [...],
            "background_info": "患者三年前进行鼻中隔手术，术后长期存在后鼻滴漏和不适感",
            "performance": [("extract", {...})],
        }
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("extract_user_info")["system"]),
        MessagesPlaceholder(variable_name="asking_history"),
        ("human", get_prompt_template("extract_user_info")["user"]),
    ])

    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "question": state["asking_messages"][-1][0].content,
        "asking_history": state["asking_messages"][-1],
    })

    state["background_info"] = ai["msg"]
    state["performance"].append(("extract", ai))
    return state


def judge_split_query(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """
    【节点 3】判断是否需要拆分多个子查询进行检索

    功能说明：
    1. 根据用户问题和背景信息，判断是否需要将问题拆分为多个子查询
    2. 如果需要拆分，生成最多 3 个相互独立的子查询
    3. 如果不需要拆分，生成一个检索友好的改写查询
    4. 将结果写入 state["sub_query"]

    输入样例：
        state = {
            "background_info": "患者三年前进行鼻中隔手术，术后长期存在后鼻滴漏和不适感",
            "asking_messages": [
                [
                    HumanMessage("我最近做了鼻中隔手术"),
                    AIMessage("请问手术多久了？是否有后鼻滴漏？"),
                    HumanMessage("三年前做的手术，一直有后鼻滴漏"),
                    AIMessage("不需要询问任何其他信息"),
                ],
            ],
            "dialogue_messages": [
                HumanMessage("鼻中隔手术后一直不舒服"),
            ],
            "multi_summary": "",
        }

    输出样例（拆分情况）：
        state.sub_query = {
            "need_split": True,
            "sub_query": [
                "鼻中隔术后 后鼻滴漏 原因",
                "鼻中隔术后 发声 影响",
                "鼻中隔术后 治疗方案",
            ],
            "rewrite_query": "",
        }

    输出样例（不拆分情况）：
        state.sub_query = {
            "need_split": False,
            "sub_query": [],
            "rewrite_query": "鼻中隔手术后三年仍有后鼻滴漏和不适感的原因及治疗方案",
        }
    """

    parser = PydanticOutputParser(pydantic_object=SplitQuery)
    fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            get_prompt_template("handle_query")["system"].format(
                format_instructions=parser.get_format_instructions()
                .replace("{", "{{")
                .replace("}", "}}"),
                summary=state["multi_summary"],
            ),
        ),
        MessagesPlaceholder(variable_name="dialogue_messages"),
        ("user", get_prompt_template("handle_query")["user"]),
    ])

    ai = (prompt | llm | RunnableLambda(strip_think_get_tokens)).invoke({
        "background_info": state["background_info"],
        "question": state["asking_messages"][-1][0].content,
        "dialogue_messages": state["dialogue_messages"],
    })

    patch: SplitQuery = fixing.parse(ai["msg"])
    patch.sub_query = patch.sub_query[:3]  # 最多 3 个子查询

    state["sub_query"] = patch
    state["performance"].append(("split_query", ai))
    return state


def run_parallel_subgraphs(state: MedicalAgentState, search_graph: SearchGraph) -> MedicalAgentState:
    """
    【节点 4】并行执行多个 SearchGraph 子图

    功能说明：
    1. 根据子查询规划结果，构建任务列表（拆分则多任务，不拆分则单任务）
    2. 为每个子查询创建独立的 SearchMessagesState
    3. 使用线程池并行执行所有 SearchGraph 子图
    4. 收集所有子查询的执行结果到 state["sub_query_results"]

    输入样例：
        state = {
            "sub_query": {
                "need_split": True,
                "sub_query": [
                    "鼻中隔术后 后鼻滴漏 原因",
                    "鼻中隔术后 发声 影响",
                ],
                "rewrite_query": "",
            },
            "sub_query_results": [],
        }

    输出样例：
        state = {
            "sub_query": {...},
            "sub_query_results": [
                {
                    "query": "鼻中隔术后 后鼻滴漏 原因",
                    "main_messages": [...],
                    "other_messages": [],
                    "docs": [doc1, doc2, ...],
                    "summary": "鼻中隔术后后鼻滴漏的常见原因包括...",
                    "retry": 2,
                    "final": "鼻中隔术后后鼻滴漏...",
                },
                {
                    "query": "鼻中隔术后 发声 影响",
                    "main_messages": [...],
                    "other_messages": [],
                    "docs": [doc3, doc4, ...],
                    "summary": "鼻中隔手术对发声的影响...",
                    "retry": 2,
                    "final": "鼻中隔手术对发声的影响...",
                },
            ],
        }
    """

    # ---------- 步骤1：构建待执行的任务列表 ----------
    sq = state.get("sub_query")
    # 如果需要拆分，则使用多个子查询；否则使用单个改写后的查询
    # jobs 最终是一个查询字符串列表，如 ["查询1", "查询2"] 或 ["改写后查询"]
    jobs = sq.sub_query if sq and sq.need_split else [sq.rewrite_query]

    # ---------- 步骤2：定义单个查询的执行函数 ----------
    def _run_one(query: str) -> SearchMessagesState:
        """
        为单个查询创建初始 State 并运行 SearchGraph

        SearchMessagesState 是 SearchGraph 子图的状态类型，包含：
        - query: 当前查询
        - main_messages: 主对话消息
        - other_messages: 其他消息
        - docs: 检索到的文档
        - summary: 摘要
        - retry: 剩余重试次数
        - final: 最终答案
        """
        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "summary": "",
            "retry": search_graph.config.agent.max_attempts,
            "final": "",
        }
        return search_graph.run(init_state)

    # ---------- 步骤3：使用线程池并行执行所有查询 ----------
    results: List[SearchMessagesState] = []  # 存储所有子查询的执行结果
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        # 提交所有任务到线程池，每个查询对应一个 Future 对象
        futures = [ex.submit(_run_one, q) for q in jobs]
        # 按完成顺序收集结果（as_completed 返回最先完成的 Future）
        for f in as_completed(futures):
            results.append(f.result())

    # ---------- 步骤4：将所有结果写入主 State ----------
    state["sub_query_results"] = results
    return state


def gather_answer(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """
    【节点 5】汇总所有子查询结果，写入对话历史

    功能说明：
    1. 重置追问计数器，准备接收下一轮用户提问
    2. 将当前用户输入保存到主对话历史
    3. 汇总所有并行子查询的检索结果（summary 字段）
    4. 将汇总结果作为 AI 回复写入对话历史
    5. 设置 final_answer 供前端 UI 使用

    输入样例：
        state = {
            "curr_ask_num": 2,
            "curr_input": "鼻中隔手术后一直不舒服",
            "dialogue_messages": [
                HumanMessage("我最近做了鼻中隔手术"),
                AIMessage("请问手术多久了？"),
                HumanMessage("大概一周了"),
                AIMessage("有什么具体症状吗？"),
            ],
            "sub_query_results": [
                {"summary": "鼻中隔术后一周内常见不适症状包括鼻塞、鼻干、轻微疼痛...", "sources": [...]},
                {"summary": "术后护理建议：避免用力擤鼻，保持鼻腔湿润...", "sources": [...]},
            ],
        }

    输出样例：
        state = {
            "curr_ask_num": 0,
            "curr_input": "鼻中隔手术后一直不舒服",
            "dialogue_messages": [
                HumanMessage("我最近做了鼻中隔手术"),
                AIMessage("请问手术多久了？"),
                HumanMessage("大概一周了"),
                AIMessage("有什么具体症状吗？"),
                HumanMessage("鼻中隔手术后一直不舒服"),
                AIMessage("鼻中隔术后一周内常见不适症状包括鼻塞、鼻干、轻微疼痛...\n术后护理建议：避免用力擤鼻，保持鼻腔湿润..."),
            ],
            "sub_query_results": [...],
            "final_answer": "鼻中隔术后一周内常见不适症状包括鼻塞、鼻干、轻微疼痛...\n术后护理建议：避免用力擤鼻，保持鼻腔湿润...",
        }
    """

    # 重置追问计数器，为下一轮对话做准备
    state["curr_ask_num"] = 0

    # 将当前用户输入追加到主对话历史
    state["dialogue_messages"].append(
        HumanMessage(content=state["curr_input"])
    )

    # 从所有子查询结果中提取 summary 字段，用换行符拼接成完整答案
    summary_text = "\n".join([
        item.get("summary", "") for item in state.get("sub_query_results", [])
    ])

    # 将汇总结果作为 AI 回复写入对话历史
    state["dialogue_messages"].append(
        AIMessage(content=summary_text)
    )

    # 设置最终答案字段（供前端 UI 直接读取）
    state["final_answer"] = summary_text
    return state


# =========================================================
# 四、MedicalAgent 主类（对外接口）
# =========================================================

class MedicalAgent:
    """
    MedicalAgent：对外暴露的统一问答入口
    """

    def __init__(self, config: AppConfig, power_model: BaseChatModel) -> None:
        self.config = config
        self.power_model = power_model
        self.normal_llm = create_llm_client(self.config.llm)
        self.search_graph = SearchGraph(self.config, power_model)
        self._build_graph()

    def _build_graph(self):
        g = StateGraph(MedicalAgentState)

        g.add_node("ask", partial(ask_judge, llm=self.normal_llm))
        g.add_node("extract", partial(extract_background_info, llm=self.normal_llm))
        g.add_node("split", partial(judge_split_query, llm=self.power_model))
        g.add_node("run", partial(run_parallel_subgraphs, search_graph=self.search_graph))
        g.add_node("answer", partial(gather_answer, llm=self.normal_llm))

        g.set_entry_point("ask")
        g.add_conditional_edges("ask", route_ask_again, {"ask": END, "pass": "extract"})
        g.add_edge("extract", "split")
        g.add_edge("split", "run")
        g.add_edge("run", "answer")
        g.add_edge("answer", END)

        self.app = g.compile()

        # 初始化 State
        self.state: MedicalAgentState = {
            "dialogue_messages": [],
            "asking_messages": [],
            "background_info": "",
            "multi_summary": [],
            "curr_input": "",
            "sub_query_results": [],
            "max_ask_num": 5,
            "curr_ask_num": 0,
            "final_answer": "",
            "performance": [],
        }

    def answer(self, user_input: str) -> MedicalAgentState:
        """
        对外统一调用入口

        使用示例：
            agent.answer("鼻中隔手术后一直不舒服怎么办？")
        """
        self.state["curr_input"] = user_input
        self.state = self.app.invoke(self.state)
        return self.state
