"""多轮对话Agent：多轮医疗对话 + 规划式 RAG Agent"""
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Any

from langchain.output_parsers import OutputFixingParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# 添加当前项目目录到 Python 路径（支持直接运行）
project_dir = Path(__file__).parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# 导入配置和工具函数
try:
    # 尝试相对导入（当作为包导入时）
    from ..config.models import AppConfig
    from .utils import create_llm_client
    from agent.templates import get_prompt_template
    from .utils import strip_think_get_tokens
except ImportError:
    # 回退到直接导入（当直接运行文件时）
    from config.models import AppConfig
    from agent.utils import create_llm_client
    from prompts.templates import get_prompt_template
    from agent.utils import strip_think_get_tokens

# 导入 SearchMessagesState 和 SearchGraph（类型注解需要，使用 TYPE_CHECKING）
if TYPE_CHECKING:
    from .search_graph import SearchGraph, SearchMessagesState
else:
    # 运行时只导入 SearchMessagesState（用于运行时类型检查）
    try:
        from .search_graph import SearchMessagesState
    except ImportError:
        from search_graph import SearchMessagesState

logger = logging.getLogger(__name__)


# =============================================================================
# 结构化模型定义（LLM 输出约束）
# =============================================================================
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


class SplitQuery(BaseModel):
    """LLM 用于判断是否需要拆分多个子查询的结构"""
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


# =============================================================================
# LangGraph 顶层 State 定义
# =============================================================================
class MedicalAgentState(TypedDict, total=False):
    """MedicalAgent 在 LangGraph 中流转的【唯一状态对象】"""

    # ---------- 对话与上下文 ----------
    dialogue_messages: List[BaseMessage]  # 主对话历史
    asking_messages: List[List[BaseMessage]]  # 每一轮追问形成一组子对话
    background_info: str  # 从追问中抽取的背景摘要
    curr_input: str  # 当前用户输入
    multi_summary: List[str]  # 多轮对话摘要列表

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


# =============================================================================
# 节点函数定义
# =============================================================================
def ask_judge(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """
    【节点 1】判断是否需要向用户继续追问关键信息

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


def route_ask_again(state: MedicalAgentState) -> str:
    """路由函数：判断是否需要继续向用户追问"""
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


def judge_split_query(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """
    【节点 3】判断是否需要拆分多个子查询进行检索

    功能说明：
    1. 根据用户问题和背景信息，判断是否需要将问题拆分为多个子查询
    2. 如果需要拆分，生成最多 3 个相互独立的子查询
    3. 如果不需要拆分，生成一个检索友好的改写查询
    4. 将结果写入 state["sub_query"]
    """
    # ========================================================
    # 步骤 1: 创建解析器（将 LLM 输出解析为 SplitQuery 结构化对象）
    # ========================================================
    parser = PydanticOutputParser(pydantic_object=SplitQuery)
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
            get_prompt_template("handle_query")["system"].format(
                format_instructions=parser.get_format_instructions()
                .replace("{", "{{")
                .replace("}", "}}"),
                summary=multi_summary_text,
            ),
        ),
        # 插入主对话历史（不含追问过程的完整对话）
        MessagesPlaceholder(variable_name="dialogue_messages"),
        # 用户消息：背景信息和当前问题
        ("user", get_prompt_template("handle_query")["user"]),
    ])

    # ========================================================
    # 步骤 4: 构建执行链并调用 LLM
    # ========================================================
    # 执行链示例:
    #   prompt（模板）→ 填充变量 → llm（生成）→ strip_think_get_tokens（清理）→ 返回字典
    #
    # 实际执行示例:
    #   输入数据:
    #     background_info = "患者主诉头痛，持续3天，伴有低烧症状"
    #     question = "怎么治疗？"
    #     dialogue_messages = [
    #       HumanMessage(content="我头痛"),
    #       AIMessage(content="患者主诉头痛，持续3天，伴有低烧症状。建议您先测量体温...")
    #     ]
    #
    #   prompt 格式化后发送给 LLM
    #   系统提示: "你是一名资深的医务人员。任务：根据历史对话、摘要、以及所掌握的用户信息..."
    #   JSON schema: {"properties": {"need_split": {...}, "sub_query": {...}, "rewrite_query": {...}}, ...}
    #   对话摘要: "用户询问头痛问题\n用户提到持续三天且有低烧"
    #   对话历史: [Human:我头痛, AI:患者主诉头痛...]
    #   用户消息: "基本背景信息: 患者主诉头痛，持续3天，伴有低烧症状\n用户当前提问: 怎么治疗？"
    #
    #   ↓ LLM 返回 AIMessage
    #   AIMessage(content='{\n  "need_split": true,\n  "sub_query": ["头痛伴随低烧的诊断标准", "头痛伴低烧的治疗方案", "头痛低烧的注意事项"],\n  "rewrite_query": ""\n}')
    #
    #   ↓ strip_think_get_tokens 处理
    #   ai = {
    #     "msg": '{\n  "need_split": true,\n  "sub_query": ["头痛伴随低烧的诊断标准", "头痛伴低烧的治疗方案", "头痛低烧的注意事项"],\n  "rewrite_query": ""\n}',
    #     "msg_len": 150,
    #     "msg_token_len": 80,
    #     "generate_time": 1.2
    #   }
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
        logger.warning(f"LLM 返回空结果，使用默认值: need_split=False, sub_query=[], rewrite_query=原始问题")
        patch = SplitQuery(need_split=False, sub_query=[], rewrite_query=state["asking_messages"][-1][0].content)
    else:
        try:
            # 解析原理同 ask_judge 节点：
            # 1) Prompt 中的 format_instructions 提供 SplitQuery 的 JSON schema
            # 2) LLM 根据 schema 生成结构化 JSON（如: {"need_split": true, "sub_query": [...], "rewrite_query": ""}）
            # 3) OutputFixingParser 容错解析，自动修复格式错误
            # 4) 最终得到 patch = SplitQuery(need_split=True, sub_query=["..."], rewrite_query="")
            patch: SplitQuery = fixing.parse(ai["msg"])
        except Exception as e:
            logger.error(f"解析 LLM 输出失败: {e}, 输出内容: {ai['msg'][:200]}")
            # 解析失败时使用默认值
            patch = SplitQuery(need_split=False, sub_query=[], rewrite_query=state["asking_messages"][-1][0].content)

    # ========================================================
    # 步骤 6: 限制子查询数量并更新状态
    # ========================================================
    # 确保最多只保留 3 个子查询（即使 LLM 生成更多）
    # 示例: patch.sub_query 从 ["诊断标准", "治疗方案", "注意事项", "预防措施"]
    #       → ["诊断标准", "治疗方案", "注意事项"]
    patch.sub_query = patch.sub_query[:3]

    # 保存查询规划结果
    # 示例结果: state["sub_query"] = SplitQuery(need_split=True, sub_query=["头痛伴随低烧的诊断标准", "头痛伴低烧的治疗方案", "头痛低烧的注意事项"], rewrite_query="")
    state["sub_query"] = patch
    state["performance"].append(("split_query", ai))
    return state


def run_parallel_subgraphs(state: MedicalAgentState, search_graph: "SearchGraph") -> MedicalAgentState:
    """
    【节点 4】并行执行多个 SearchGraph 子图

    功能说明：
    1. 根据子查询规划结果，构建任务列表
    2. 为每个子查询创建独立的 SearchMessagesState
    3. 使用线程池并行执行所有 SearchGraph 子图
    4. 收集所有子查询的执行结果到 state["sub_query_results"]
    """
    sq = state.get("sub_query")
    jobs = sq.sub_query if sq and sq.need_split else [sq.rewrite_query]

    def _run_one(query: str) -> SearchMessagesState:
        """为单个查询创建初始 State 并运行 SearchGraph"""
        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "answer": "",
            "retry": search_graph.config.agent.max_attempts,
            "final": "",
        }
        return search_graph.run(init_state)

    results: List[SearchMessagesState] = []
    if jobs:  # 只有在有任务时才创建线程池
        with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
            futures = [ex.submit(_run_one, q) for q in jobs]
            for f in as_completed(futures):
                results.append(f.result())

    state["sub_query_results"] = results
    return state


def gather_answer(state: MedicalAgentState, llm: BaseChatModel) -> MedicalAgentState:
    """
    【节点 5】汇总所有子查询结果，写入对话历史

    功能说明：
    1. 重置追问计数器，准备接收下一轮用户提问
    2. 将当前用户输入保存到主对话历史
    3. 汇总所有并行子查询的检索结果
    4. 将汇总结果作为 AI 回复写入对话历史
    5. 设置 final_answer 供前端 UI 使用
    """
    state["curr_ask_num"] = 0

    state["dialogue_messages"].append(
        HumanMessage(content=state["curr_input"])
    )

    summary_text = "\n".join([
        item.get("answer", "") for item in state.get("sub_query_results", [])
    ])

    state["dialogue_messages"].append(
        AIMessage(content=summary_text)
    )

    state["final_answer"] = summary_text
    return state


# =============================================================================
# 多轮对话Agent类
# =============================================================================
class MultiDialogueAgent:
    """多轮对话Agent：对外暴露的统一问答入口"""

    def __init__(self, config: AppConfig, power_model: BaseChatModel, websearch_func=None) -> None:
        """
        初始化多轮对话Agent

        Args:
            config: 应用配置
            power_model: 强大模型实例（用于工具调用）
            websearch_func: 网络搜索函数（可选）
        """
        self.config = config
        self.power_model = power_model
        self.normal_llm = create_llm_client(config.llm)

        # 延迟导入SearchGraph以避免循环依赖
        try:
            from .search_graph import SearchGraph
        except ImportError:
            from search_graph import SearchGraph

        self.search_graph = SearchGraph(config, power_model)
        self._build_graph()

    def _build_graph(self):
        """
        构建多轮对话图

        功能说明：
        创建一个包含5个节点的状态图，用于处理多轮医疗对话。
        节点顺序：ask → extract → split → run → answer

        节点功能详解：
        1. ask 节点：判断是否需要向用户追问（如症状细节、病史等），最多追问 max_ask_num 轮
        2. extract 节点：从追问对话中抽取患者背景信息（症状、病史、用药情况等）
        3. split 节点：分析问题是否需要拆分为多个子查询（如诊断、治疗、注意事项分别检索）
        4. run 节点：并行执行多个 SearchGraph 子图，进行知识检索和答案生成
        5. answer 节点：汇总子查询结果，生成最终答案并更新对话历史
        """
        g = StateGraph(MedicalAgentState)

        # 输入示例（ask节点）：
        #   state = {
        #     "background_info": "",
        #     "curr_input": "我头痛",
        #     "asking_messages": [],
        #     "curr_ask_num": 0,
        #     "max_ask_num": 3
        #   }
        g.add_node("ask", partial(ask_judge, llm=self.normal_llm))

        # 输入示例（extract节点）：
        #   state = {
        #     "asking_messages": [
        #       [HumanMessage("我头痛"),
        #        AIMessage("头痛持续多久了? 有发热吗?"),
        #        HumanMessage("三天了，有点低烧"),
        #        AIMessage("不需要询问任何其他信息")]
        #     ]
        #   }
        g.add_node("extract", partial(extract_background_info, llm=self.normal_llm))

        # 输入示例（split节点）：
        #   state = {
        #     "background_info": "患者主诉头痛，持续3天，伴有低烧症状",
        #     "asking_messages": [[HumanMessage("我头痛")]],
        #     "dialogue_messages": [],
        #     "multi_summary": []
        #   }
        g.add_node("split", partial(judge_split_query, llm=self.power_model))

        # 输入示例（run节点）：
        #   state = {
        #     "sub_query": SplitQuery(
        #       need_split=True,
        #       sub_query=["头痛伴随低烧的诊断标准", "头痛伴低烧的治疗方案", "头痛低烧的注意事项"],
        #       rewrite_query=""
        #     )
        #   }
        g.add_node("run", partial(run_parallel_subgraphs, search_graph=self.search_graph))

        # 输入示例（answer节点）：
        #   state = {
        #     "curr_input": "我头痛",
        #     "sub_query_results": [
        #       {"answer": "头痛伴低烧的常见病因包括..."},
        #       {"answer": "治疗方案建议如下..."},
        #       {"answer": "注意事项：应多休息..."}
        #     ],
        #     "dialogue_messages": []
        #   }
        g.add_node("answer", partial(gather_answer, llm=self.normal_llm))

        g.set_entry_point("ask")

        # 条件边：ask节点输出
        #   state = {"ask_obj": AskMess(need_ask=True, questions=["头痛持续多久了?"]), "curr_ask_num": 1, "max_ask_num": 3}
        #   结果：需要追问且未达上限 → 返回 "ask"，流程结束等待用户回复
        #   state = {"ask_obj": AskMess(need_ask=False), "curr_ask_num": 2, "max_ask_num": 3}
        #   结果：不需要追问 → 返回 "pass"，继续到 extract 节点
        # 如何跳转：
        #   route_ask_again 返回 "ask" → 跳转到 END（流程结束，等待用户输入）
        #   route_ask_again 返回 "pass" → 跳转到 "extract" 节点
        g.add_conditional_edges("ask", route_ask_again, {"ask": END, "pass": "extract"})

        # extract节点输出示例：
        #   state = {"background_info": "患者主诉头痛，持续3天，伴有低烧症状", "performance": [("extract", {...})]}
        g.add_edge("extract", "split")

        # split节点输出示例：
        #   state = {
        #     "sub_query": SplitQuery(
        #       need_split=True,
        #       sub_query=["头痛伴随低烧的诊断标准", "头痛伴低烧的治疗方案", "头痛低烧的注意事项"],
        #       rewrite_query=""
        #     ),
        #     "performance": [...]
        #   }
        g.add_edge("split", "run")

        # run节点输出示例：
        #   state = {
        #     "sub_query_results": [
        #       {"query": "头痛伴随低烧的诊断标准", "answer": "头痛伴低烧的常见病因包括...", "docs": [...]},
        #       {"query": "头痛伴低烧的治疗方案", "answer": "治疗方案建议如下...", "docs": [...]},
        #       {"query": "头痛低烧的注意事项", "answer": "注意事项：应多休息...", "docs": [...]}
        #     ]
        #   }
        g.add_edge("run", "answer")

        # answer节点输出示例：
        #   state = {
        #     "curr_ask_num": 0,
        #     "dialogue_messages": [
        #       HumanMessage(content="我头痛"),
        #       AIMessage(content="头痛伴低烧的常见病因包括...\n\n治疗方案建议如下...\n\n注意事项：应多休息...")
        #     ],
        #     "final_answer": "头痛伴低烧的常见病因包括...\n\n治疗方案建议如下...\n\n注意事项：应多休息..."
        #   }
        g.add_edge("answer", END)

        self.app = g.compile()

        self.state: MedicalAgentState = {
            "dialogue_messages": [],
            "asking_messages": [],
            "background_info": "",
            "multi_summary": [],
            "curr_input": "",
            "sub_query_results": [],
            "max_ask_num": self.config.agent.max_ask_num,
            "curr_ask_num": 0,
            "final_answer": "",
            "performance": [],
        }

    def answer(self, user_input: str) -> MedicalAgentState:
        """
        对外统一调用入口

        Args:
            user_input: 用户输入

        Returns:
            更新后的状态对象
        """
        self.state["curr_input"] = user_input
        self.state = self.app.invoke(self.state)
        return self.state

    def reset(self):
        """重置对话状态"""
        self.state = {
            "dialogue_messages": [],
            "asking_messages": [],
            "background_info": "",
            "multi_summary": [],
            "curr_input": "",
            "sub_query_results": [],
            "max_ask_num": self.config.agent.max_ask_num,
            "curr_ask_num": 0,
            "final_answer": "",
            "performance": [],
        }
