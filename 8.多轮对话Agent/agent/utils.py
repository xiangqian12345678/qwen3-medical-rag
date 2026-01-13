"""工具函数"""
import re
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    pass


def strip_think_get_tokens(msg: AIMessage):
    text = msg.content
    msg_len = len(msg.content)

    try:
        # 1. usage_metadata["output_tokens"]: LangChain 新版消息对象直接提供的元数据
        #   包含输入/输出的 token 使用情况
        msg_token_len = msg.usage_metadata["output_tokens"]
    except Exception as e1:
        try:
            # 2. response_metadata["token_usage"]["output_tokens"]: 底层模型响应中的 token 使用信息
            #   通过模型 API 返回的原始数据结构获取
            msg_token_len = msg.response_metadata["token_usage"]["output_tokens"]
        except Exception as e2:
            msg_token_len = 0

    # 3. response_metadata["total_duration"]: 模型生成总耗时(纳秒),除以1e9转换为秒
    dur = msg.response_metadata.get("total_duration", 0) / 1e9

    return {
        "msg": re.sub(r"</think>.*?</think>\s*", "", text, flags=re.DOTALL).strip(),
        "msg_len": msg_len,
        "msg_token_len": msg_token_len,
        "generate_time": dur
    }


def del_think(text: str) -> str:
    """移除思考过程标签"""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def json_to_list_document(text: str):
    """将JSON转换为Document列表

    Args:
        text: JSON格式的字符串,包含多个Document字典的列表

    Returns:
        Document对象列表

    Example:
        >>> json_str = '''[{"page_content": "糖尿病症状", "metadata": {"source": "doc1"}},
        ...                {"page_content": "高血压治疗", "metadata": {"source": "doc2"}}]'''
        >>> docs = json_to_list_document(json_str)
        >>> docs[0].page_content
        '糖尿病症状'
        >>> docs[0].metadata
        {'source': 'doc1'}
    """
    import json
    from langchain_core.documents import Document
    return [Document(**d) for d in json.loads(text)]


def format_document_str(documents) -> str:
    """格式化文档为字符串"""
    parts = []
    for i, d in enumerate(reversed(documents)):
        if i >= 6:
            break
        parts.append(f"## 文档{i + 1}：\n{d.page_content}\n")
    return "".join(parts)


def _should_call_tool(last_ai) -> bool:
    """判断是否需要调用工具Args:
    last_ai: 上一次AI返回的消息对象(AIMessage)

    Returns:
        bool: 如果消息包含tool_calls属性且不为空,返回True;否则返回False

    Example:
        >>> from langchain_core.messages import AIMessage
        >>> msg = AIMessage(content="查询天气")
        >>> _should_call_tool(msg)
        False
        >>> msg = AIMessage(content="", tool_calls=[{"name": "search", "args": {}}])
        >>> _should_call_tool(msg)
        True
    """
    return bool(getattr(last_ai, 'tool_calls', None))
