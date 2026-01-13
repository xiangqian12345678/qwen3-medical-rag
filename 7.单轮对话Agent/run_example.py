"""单轮对话Agent示例"""
import logging

from agent import SingleDialogueAgent
# 使用完整路径导入
from config import ConfigLoader
from core import create_llm_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.config

    logger.info(f"配置加载完成: {config.milvus.collection_name}")
    logger.info(f"Agent模式: {config.agent.mode}")
    logger.info(f"LLM模型: {config.llm.model}")

    # 创建LLM客户端（用于工具调用）
    power_llm = create_llm_client(config.llm)

    # 创建单轮对话Agent
    agent = SingleDialogueAgent(config, power_llm)

    # 测试查询
    test_queries = [
        "高血压的症状有哪些？",
        "糖尿病患者饮食需要注意什么？",
        "感冒如何预防？"
    ]

    for query in test_queries:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"用户问题: {query}")
        logger.info(f"{'=' * 50}")

        try:
            answer = agent.answer(query)
            logger.info(f"回答:\n{answer}")
        except Exception as e:
            logger.error(f"处理失败: {e}")


if __name__ == "__main__":
    main()
