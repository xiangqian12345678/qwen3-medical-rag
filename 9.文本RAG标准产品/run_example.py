"""
多轮对话Agent示例

演示如何使用多轮对话Agent进行医疗问答
"""
import logging
import os
from pathlib import Path

# 加载环境变量
from dotenv import load_dotenv
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    logging.info(f"已加载环境变量文件: {env_file}")
else:
    logging.warning(f"未找到 .env 文件,请确保 DASHSCOPE_API_KEY 环境变量已设置")

from config.loader import ConfigLoader
from agent import MultiDialogueAgent, create_llm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    """主函数"""
    # 初始化配置
    config_manager = ConfigLoader()
    config = config_manager.config

    # 创建LLM客户端
    power_model = create_llm_client(config.llm)

    # 初始化多轮对话Agent
    agent = MultiDialogueAgent(config, power_model=power_model)

    # 示例对话
    print("=" * 50)
    print("多轮对话Agent示例")
    print("=" * 50)

    try:
        # 第一轮对话
        user_input = "我这两天肚子痛，还拉肚子"
        print(f"\n用户: {user_input}")

        state = agent.answer(user_input=user_input)

        if state["ask_obj"].need_ask:
            print(f"\nAgent: {state['asking_messages'][-1][-1].content}")
            print("\n（需要追问更多信息）")

            # 模拟用户回复追问
            user_input = "肚子疼得厉害，大概从昨天开始的"
            print(f"\n用户: {user_input}")

            state = agent.answer(user_input=user_input)

        # 输出最终回答
        print(f"\nAgent: {state['final_answer']}")

        # 第二轮对话
        print("\n" + "=" * 50)
        user_input = "那应该吃什么药？"
        print(f"\n用户: {user_input}")

        state = agent.answer(user_input=user_input)
        print(f"\nAgent: {state['final_answer']}")

        # 输出性能信息
        print("\n" + "=" * 50)
        print("性能信息:")
        for name, perf in state.get("performance", []):
            print(f"  {name}: {perf}")

    except Exception as e:
        logger.error(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
