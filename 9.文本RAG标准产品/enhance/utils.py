"""工具函数"""
import logging
import re

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def strip_think_get_tokens(msg: AIMessage):
    text = msg.content
    msg_len = len(msg.content)

    try:
        msg_token_len = msg.usage_metadata["output_tokens"]
    except Exception as e1:
        try:
            msg_token_len = msg.response_metadata["token_usage"]["output_tokens"]
        except Exception as e2:
            msg_token_len = 0

    dur = msg.response_metadata.get("total_duration", 0) / 1e9

    return {
        "msg": re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip(),
        "msg_len": msg_len,
        "msg_token_len": msg_token_len,
        "generate_time": dur
    }


def del_think(text: str) -> str:
    """移除思考过程标签"""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def format_document_str(documents) -> str:
    """格式化文档为字符串"""
    parts = []
    for i, d in enumerate(reversed(documents)):
        if i >= 6:
            break
        parts.append(f"## 文档{i + 1}：\n{d.page_content}\n")
    return "".join(parts)
