"""工具函数"""
import logging
import re
import time
from typing import Any, Callable, Optional

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


def record_timing_to_state(
    stage_name: str,
    duration: float,
    state: Optional[dict] = None,
    additional_info: Optional[dict] = None
) -> None:
    """
    记录耗时信息到 state 的 performance 列表中

    Args:
        stage_name: 阶段名称
        duration: 耗时（秒）
        state: 状态对象，包含 performance 列表
        additional_info: 额外信息字典，会合并到记录中
    """
    if state is not None and "performance" in state:
        perf_info = {
            "stage": stage_name,
            "duration": duration,
            "timestamp": time.time()
        }
        if additional_info:
            perf_info.update(additional_info)
        state["performance"].append(perf_info)

    # 记录日志
    logger.info(f"  {stage_name}: {duration:.2f}秒")
