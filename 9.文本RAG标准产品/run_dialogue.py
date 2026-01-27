"""
多轮对话Agent示例

演示如何使用多轮对话Agent进行医疗问答
"""
import logging

from app_config import APPConfig
from recall.kgraph.kg_loader import KGraphConfigLoader
from recall.milvus.embed_loader import EmbedConfigLoader
from dialogue_agent import DialogueAgent
from rag_loader import RAGConfigLoader
from utils import create_llm_client, create_embedding_client, create_reranker_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    """主函数"""
    # 初始化配置
    rag_config_loader = RAGConfigLoader()  # 与milvus_config 和 kgraph_config 的loader不同
    milvus_config_loader = EmbedConfigLoader()
    kgraph_config_loader = KGraphConfigLoader()
    app_config = APPConfig(rag_config_loader=rag_config_loader,
                           milvus_config_loader=milvus_config_loader,
                           kgraph_config_loader=kgraph_config_loader)
    rag_config = rag_config_loader.config

    # 创建LLM客户端
    llm_model = create_llm_client(rag_config.llm)
    embed_model = create_embedding_client(rag_config.embedding)
    reranker = create_reranker_client(rag_config.reranker)

    # 初始化多轮对话Agent
    agent = DialogueAgent(app_config=app_config, embeddings_model=embed_model, llm=llm_model, reranker=reranker)

    # 交互式对话
    print("=" * 50)
    print("多轮对话Agent - 交互模式")
    print("=" * 50)
    print("输入 'exit' 或 'quit' 退出对话\n")

    # user_input = "什么是房颤？"  会触发所有的召回
    try:
        while True:
            # 获取用户输入
            user_input = input("用户: ").strip()

            # 检查退出命令
            if user_input.lower() in ['exit', 'quit', '退出', 'q']:
                print("\n退出对话...")
                break

            # 跳过空输入
            if not user_input:
                continue

            # 调用Agent
            state = agent.answer(query=user_input)

            # 处理追问 - 使用while循环支持多次追问
            while state["ask_obj"].need_ask:
                print(f"Agent: {state['asking_messages'][-1][-1].content}")
                print("（需要追问更多信息）")

                # 获取用户回复
                user_input = input("用户: ").strip()

                # 检查退出命令
                if user_input.lower() in ['exit', 'quit', '退出', 'q']:
                    return

                # 跳过空输入
                if not user_input:
                    continue

                # 继续调用Agent
                state = agent.answer(query=user_input)

            # 输出最终回答
            print(f"\nAgent: {state['final_answer']}")

            # 输出性能信息
            print("\n" + "-" * 50)
            print("性能信息:")
            for name, perf in state.get("performance", []):
                print(f"  {name}: {perf}")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n\n用户中断，退出对话...")
    except Exception as e:
        logger.error(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
