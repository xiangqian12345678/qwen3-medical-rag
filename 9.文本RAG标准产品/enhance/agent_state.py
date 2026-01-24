"""多轮对话Agent：多轮医疗对话 + 规划式 RAG Agent"""
from typing import List, Any

from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage
)
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from typing_extensions import TypedDict

from enhance.query_enhance import RewriteQuery
from enhance.recall_enhance import MultiQueries, SubQueries, SuperordinateQuery, HypotheticalAnswer
from search_graph import SearchMessagesState


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


class AgentState(TypedDict, total=False):
    """MedicalAgent 在 LangGraph 中流转的【唯一状态对象】"""

    # ---------- 对话与上下文 ----------
    dialogue_messages: List[BaseMessage]  # 主对话历史
    asking_messages: List[List[BaseMessage]]  # 每一轮追问形成一组子对话
    background_info: str  # 从追问中抽取的背景摘要
    curr_input: str  # 当前用户输入
    multi_summary: List[str]  # 多轮对话摘要列表

    ask_obj: AskMess  # 是否需要继续追问

    # ---------- curr_input生成 ----------
    query_results: List[Document]

    # ---------- 子问题生成 ----------
    sub_query: SubQueries  # 子查询规划结果
    sub_query_results: List[List[Document]]  # 子查询执行结果

    # ---------- 问题重构 ----------
    rewrite_query: RewriteQuery  # 并行问题生成
    rewrite_query_docs: List[Document]  # 改写问题执行结果

    # ---------- 多问题生成 ----------
    multi_query: MultiQueries  # 并行问题生成
    multi_query_docs: List[List[Document]]  # 并行问题执行结果

    # ---------- 上位问题生成 ----------
    superordinate_query: SuperordinateQuery  # 上位问题生成
    superordinate_query_docs: List[Document]  # 上位问题执行结果

    # ---------- 假设性回答 ----------
    hypothetical_answer: HypotheticalAnswer  # 假设性回答
    hypothetical_answer_docs: List[Document]  # 假设性回答执行结果

    # ----------  ----------
    multi_dialogue_results: List[SearchMessagesState]  # 多轮对话执行结果

    # ---------- 控制变量 ----------
    max_ask_num: int  # 最大追问轮次
    curr_ask_num: int  # 当前已追问次数

    # ---------- 输出 & 调试 ----------
    final_answer: str  # 最终答案（可供 UI 使用）
    performance: List[Any]  # 调试 / 性能信息
