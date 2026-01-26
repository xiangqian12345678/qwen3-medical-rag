"""RAG回答生成模块"""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .templates import get_prompt_template
from .utils import format_document_str, del_think

from typing_extensions import TypedDict


class AnswerState(TypedDict, total=False):
    query: str
    other_messages: List
    docs: List[Document]
    answer: str

logger = logging.getLogger(__name__)


def generate_answer(
        state: "AnswerState",
        llm: BaseChatModel,
        show_debug: bool
) -> "AnswerState":
    if show_debug:
        logger.info(f"开始RAG...当前文档数量: {len(state.get('docs', []))}")

    sys = get_prompt_template("basic_rag")["system"]
    user = get_prompt_template("basic_rag")["user"]

    prompt = [
        SystemMessage(content=sys),
        HumanMessage(
            content=user.format(
                all_document_str=format_document_str(state.get("docs", [])),
                input=state["query"]
            )
        )
    ]

    rag_response = llm.invoke(prompt)

    # 调试：记录 LLM 原始输出
    if show_debug:
        logger.info(f"LLM原始输出长度: {len(rag_response.content)}")
        logger.info(f"LLM原始输出(前200字符): {rag_response.content[:200]}")

    rag_content = del_think(rag_response.content)

    # 调试：记录 del_think 后的输出
    if show_debug:
        logger.info(f"del_think后长度: {len(rag_content)}")
        logger.info(f"del_think后内容(前200字符): {rag_content[:200]}")

    rag_ai = AIMessage(content=rag_content)

    if show_debug:
        logger.info(f"RAG回答: {rag_ai.content}")

    if not isinstance(state["main_messages"][-1], AIMessage):
        state["main_messages"].append(rag_ai)
    else:
        state["main_messages"].pop()
        state["main_messages"].append(rag_ai)

    state["answer"] = rag_ai.content
    return state


def judge_node(
        state: "AnswerState",
        llm: BaseChatModel,
        show_debug: bool
) -> "AnswerState":
    """质量判断节点"""
    if show_debug:
        logger.info(f"开始评估...")

    judge_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("judge_rag")["system"]),
        HumanMessage(
            content=get_prompt_template("judge_rag")["user"].format(
                format_document_str=format_document_str(state.get('docs', [])),
                query=state['query'],
                summary=state.get('answer', '')
            )
        )
    ])

    result = del_think(judge_ai.content or "").strip().lower()

    if show_debug:
        logger.info(f"评估结果: {result[:20]}")

    state["other_messages"].append(AIMessage(content=f"[JUDGE]={result}"))

    if 'y' in result:
        state["judge_result"] = "pass"
    else:
        retries_left = int(state.get("retry", 0))
        if retries_left > 0:
            state["retry"] = retries_left - 1
            state["judge_result"] = "retry"
        else:
            state["judge_result"] = "fail"

    return state


def finish_success(state: "AnswerState") -> "AnswerState":
    """成功结束节点"""
    state["final"] = (state.get("answer", "") or "").strip() or "（空）"
    return state


def finish_fail(state: "AnswerState") -> "AnswerState":
    """失败结束节点"""
    base = (state.get("answer", "") or "").strip() or "（空）"
    state["final"] = base + "\n\n（内容可能不属实）"
    return state
