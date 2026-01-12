"""基础RAG示例"""
import logging
from rag import SimpleRAG
from config.loader import ConfigLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    # 方式1: 使用配置文件
    config_loader = ConfigLoader()
    rag = SimpleRAG(config=config_loader.config)

    # 方式2: 直接指定配置路径
    # rag = SimpleRAG(config_path="simple_rag.yaml")667

    # 查询示例
    query = "我有点肚子痛，该怎么办？"

    # 执行查询（返回文档）
    result = rag.answer(query, return_document=True)

    print(f"\n{'='*60}")
    print(f"检索完成，检索用时：{result['search_time']:.3f} s，生成用时：{result['generation_time']:.3f} s")
    print(f"{'='*60}\n")
    print(f"回答：\n{result['answer']}\n")

    # 显示参考资料
    if "documents" in result:
        print(f"\n{'='*60}")
        print(f"参考资料 ({len(result['documents'])} 条)，展示前3条:")
        print(f"{'='*60}\n")
        for i, doc in enumerate(result['documents'][:3], 1):
            print(f"{i}. 来源：{doc.metadata.get('source_name', 'N/A')} | 距离：{doc.metadata.get('distance', 'N/A'):.4f}")
            content = doc.page_content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   内容：{content}\n")


if __name__ == "__main__":
    main()
