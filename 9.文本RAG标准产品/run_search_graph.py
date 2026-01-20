#!/usr/bin/env python3
"""直接运行 search_graph.py 的包装脚本"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# 添加项目目录到 Python 路径
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# 添加 agent 目录到 Python 路径
agent_dir = Path(__file__).parent / "agent"
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

# 运行 search_graph.py
import agent.search_graph

if __name__ == "__main__":
    # 直接运行 search_graph 的 main 代码
    agent.search_graph.main()
