"""
多轮对话RAG功能演示
"""
import logging
from config import ConfigLoader
from multi_dialogue_rag import MultiDialogueRag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def print_output(result):
    """打印输出结果"""
    print(f"\n检索完成，检索用时：{result['search_time']} s，"
          f"\n重写查询生成用时：{result['rewriten_generate_time']} s，"
          f"\n增强生成用时：{result['out_generate_time']} s "
          f"\n\n回答: {result['answer']}")
    # 显示参考资料
    if "documents" in result:
        print(f"\n参考资料 ({len(result['documents'])} 条)，展示前3条:\n\n")
        for i, ctx in enumerate(result['documents'][:3], 1):
            print(f"{i}. 数据源： {ctx.metadata.get('source')} "
                  f"\n数据源名：{ctx.metadata.get('source_name')} "
                  f"\n向量距离：{ctx.metadata.get('distance')}")
            content = ctx.page_content[:200] + "..." if len(ctx.page_content) > 200 else ctx.page_content
            print(f"\n\ncontent: {content}")


def main():
    # 加载配置
    config_loader = ConfigLoader()

    # 创建多轮对话RAG系统
    rag = MultiDialogueRag(config_loader.config)

    print("多轮对话医疗助手已启动（输入 'exit' 退出）")
    print("=" * 50)

    session_id = "user_001"

    while True:
        query = input("\n请输入问题：")
        if query.lower() in ['exit', 'quit', '退出']:
            break

        result = rag.answer(query, session_id=session_id, return_document=True)
        print_output(result=result)
        print("-" * 50)


if __name__ == "__main__":
    main()
