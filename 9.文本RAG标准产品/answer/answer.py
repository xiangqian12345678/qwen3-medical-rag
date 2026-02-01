"""RAG回答生成模块"""
import logging
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict

from .answer_templates import get_prompt_template
from .utils import format_document_str, del_think


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

    # 计时并调用 LLM
    start_time = time.time()
    rag_response = llm.invoke(prompt)
    generate_time = time.time() - start_time

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

    # 记录性能信息
    msg_len = len(rag_response.content)
    try:
        msg_token_len = rag_response.usage_metadata["output_tokens"]
    except Exception:
        try:
            msg_token_len = rag_response.response_metadata["token_usage"]["output_tokens"]
        except Exception:
            msg_token_len = 0

    performance_info = {
        "msg": rag_content,
        "msg_len": msg_len,
        "msg_token_len": msg_token_len,
        "generate_time": generate_time
    }

    if show_debug:
        logger.info(f"  answer: {performance_info}")

    state["answer"] = rag_ai.content
    return state
