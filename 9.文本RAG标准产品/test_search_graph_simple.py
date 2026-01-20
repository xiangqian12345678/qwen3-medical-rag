#!/usr/bin/env python3
"""简单的测试脚本：直接导入 search_graph 并验证关键组件"""
import sys
from pathlib import Path

# 设置项目路径
project_dir = Path(__file__).parent.parent.parent  # 回到 9.文本RAG标准产品 的父目录
project_dir = Path(__file__).parent  # 当前目录就是 9.文本RAG标准产品

print(f"当前工作目录: {Path.cwd()}")
print(f"项目目录: {project_dir}")
print(f"__file__: {__file__}")
print()

# 添加项目目录到 Python 路径
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# 添加 agent 目录到 Python 路径
agent_dir = project_dir / "agent"
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

print(f"sys.path 前5项:")
for i, p in enumerate(sys.path[:5], 1):
    print(f"  {i}. {p}")
print()

print("=" * 60)
print("测试导入 search_graph.py")
print("=" * 60)

try:
    print("\n[导入] search_graph...")
    import search_graph
    print("✓ search_graph 导入成功")

    # 检查 SearchGraph 类是否存在
    print("\n[检查] SearchGraph 类...")
    if hasattr(search_graph, 'SearchGraph'):
        print("✓ SearchGraph 类存在")
    else:
        print("✗ SearchGraph 类不存在")

    # 检查 SearchMessagesState 是否存在
    print("\n[检查] SearchMessagesState 类...")
    if hasattr(search_graph, 'SearchMessagesState'):
        print("✓ SearchMessagesState 类存在")
    else:
        print("✗ SearchMessagesState 类不存在")

    print("\n" + "=" * 60)
    print("测试成功！")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
