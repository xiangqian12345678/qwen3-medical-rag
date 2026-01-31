"""向量检索工具函数"""
import json
import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    pass


def json_to_list_document(text: str):
    """将JSON转换为Document列表"""
    # 检查输入是否为空
    if not text or not text.strip():
        logger.warning("json_to_list_document: 输入为空字符串")
        return []

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            logger.warning(f"json_to_list_document: 期望列表,得到 {type(data)}")
            return []
        return [Document(**d) for d in data]
    except json.JSONDecodeError as e:
        logger.error(f"json_to_list_document: JSON解析失败 - {e}")
        logger.error(f"输入内容: {text[:500] if len(text) > 500 else text}")
        return []
    except Exception as e:
        logger.error(f"json_to_list_document: 转换失败 - {e}")
        return []


def _should_call_tool(last_ai: AIMessage) -> bool:
    """判断是否需要调用工具"""
    return bool(getattr(last_ai, 'tool_calls', None))
