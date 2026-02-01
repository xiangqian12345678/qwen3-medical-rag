"""工具函数"""
import logging
import re
import time

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def strip_think_get_tokens(msg: AIMessage, generate_time: float = None):
    """
    处理 LLM 返回的消息，提取内容并统计信息

    Args:
        msg: LLM 返回的 AIMessage
        generate_time: 可选的生成时间（秒），如果未提供则尝试从 response_metadata 获取

    Returns:
        包含处理后的消息和统计信息的字典
    """
    text = msg.content
    msg_len = len(msg.content)

    try:
        msg_token_len = msg.usage_metadata["output_tokens"]
    except Exception:
        try:
            msg_token_len = msg.response_metadata["token_usage"]["output_tokens"]
        except Exception:
            msg_token_len = 0

    # 优先使用传入的生成时间，否则尝试从 response_metadata 获取
    if generate_time is not None:
        dur = generate_time
    else:
        dur = msg.response_metadata.get("total_duration", 0) / 1e9

    return {
        "msg": re.sub(r"<思考>.*?</思考>\s*", "", text, flags=re.DOTALL).strip(),
        "msg_len": msg_len,
        "msg_token_len": msg_token_len,
        "generate_time": dur
    }


def del_think(text: str) -> str:
    """移除思考过程标签"""
    return re.sub(r"<思考>.*?</思考>\s*", "", text, flags=re.DOTALL).strip()


def format_document_str(documents) -> str:
    """格式化文档为字符串"""
    parts = []
    for i, d in enumerate(reversed(documents)):
        if i >= 6:
            break
        parts.append(f"## 文档{i + 1}：\n{d.page_content}\n")
    return "".join(parts)


def invoke_with_timing(llm_chain, inputs: dict, stage_name: str = "llm_call"):
    """
    带计时和性能日志的 LLM 调用包装函数

    Args:
        llm_chain: LLM 调用链（如 prompt | llm）
        inputs: 输入参数
        stage_name: 阶段名称（用于日志记录）

    Returns:
        包含处理结果的字典，格式与 strip_think_get_tokens 返回值相同
    """
    start_time = time.time()
    ai_msg = llm_chain.invoke(inputs)
    generate_time = time.time() - start_time

    result = strip_think_get_tokens(ai_msg, generate_time=generate_time)

    # 记录性能日志
    logger.info(f"  {stage_name}: {result}")

    return result
