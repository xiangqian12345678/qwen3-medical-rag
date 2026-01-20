#!/usr/bin/env python3
"""
启动脚本：运行 search_graph 的示例

使用方法：
    python run_search_graph_example.py
"""
import sys
from pathlib import Path

# 加载环境变量
from dotenv import load_dotenv
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# 添加项目目录到 Python 路径
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# 添加 agent 目录到 Python 路径
agent_dir = project_dir / "agent"
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

print(f"项目目录: {project_dir}")
print(f"Agent目录: {agent_dir}")
print("=" * 60)
print()

try:
    # 导入必要的模块
    import logging
    from config.loader import ConfigLoader
    from agent import create_llm_client
    from agent.search_graph import SearchGraph

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.info("开始运行 SearchGraph 示例")

    # 1. 加载配置
    logger.info("[1/4] 加载配置...")
    config_manager = ConfigLoader()
    config = config_manager.config
    logger.info(f"配置加载成功!")
    logger.info(f"  - Milvus: {config.milvus.uri}")
    logger.info(f"  - Collection: {config.milvus.collection_name}")
    logger.info(f"  - LLM: {config.llm.model} ({config.llm.provider})")

    # 2. 创建LLM客户端
    logger.info("\n[2/4] 创建LLM客户端...")
    power_model = create_llm_client(config.llm)
    logger.info(f"LLM客户端创建成功!")

    # 3. 初始化SearchGraph
    logger.info("\n[3/4] 初始化SearchGraph...")
    search_graph = SearchGraph(config, power_model=power_model)
    search_graph.build_search_graph()
    logger.info("SearchGraph初始化成功!")

    # 4. 执行示例查询
    logger.info("\n[4/4] 执行示例查询...")
    example_queries = [
        "阿司匹林有哪些副作用？",
        "糖尿病的主要症状是什么？",
        "高血压患者应该注意什么？"
    ]
    query = example_queries[0]
    logger.info(f"查询: {query}")

    result = search_graph.answer(query)

    logger.info("\n查询结果:")
    logger.info("=" * 60)
    print(f"\n问题: {query}\n")
    print(f"回答:\n{result}\n")
    logger.info("=" * 60)
    logger.info("查询完成!")

except Exception as e:
    logger.error(f"\n执行失败: {e}", exc_info=True)
    sys.exit(1)
