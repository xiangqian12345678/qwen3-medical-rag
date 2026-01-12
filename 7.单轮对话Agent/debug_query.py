"""调试脚本 - 测试单个查询"""
import logging
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import ConfigLoader
from agent import SingleDialogueAgent
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

    # 测试单个查询
    query = "糖尿病患者饮食需要注意什么？"

    logger.info(f"\n{'='*60}")
    logger.info(f"用户问题: {query}")
    logger.info(f"{'='*60}\n")

    try:
        answer = agent.answer(query)
        logger.info(f"\n{'='*60}")
        logger.info(f"回答:\n{answer}")
        logger.info(f"{'='*60}")
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
