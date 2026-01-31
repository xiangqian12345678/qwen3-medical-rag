"""工具函数"""
import logging
import re

logger = logging.getLogger(__name__)


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
