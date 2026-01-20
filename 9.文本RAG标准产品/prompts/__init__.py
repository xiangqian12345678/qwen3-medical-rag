"""Prompt模板包"""
import sys
from pathlib import Path

# 添加当前目录到 Python 路径（支持直接运行）
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    # 尝试相对导入（当作为包导入时）
    from .templates import get_prompt_template, register_prompt_template
except ImportError:
    # 回退到直接导入（当直接运行文件时）
    from templates import get_prompt_template, register_prompt_template

__all__ = ['get_prompt_template', 'register_prompt_template']
