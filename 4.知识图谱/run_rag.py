"""
RAG系统调试脚本
用于诊断为什么RAG查询找不到实体
"""
from neo4j_connection import Neo4jConnection
from rag_system import RAGSystem


def main():
    """
        示例6: 完整的RAG单轮会话
        演示从用户问题到生成答案的完整流程
        """
    print("\n" + "=" * 60)
    print("示例6: 完整的RAG单轮会话")
    print("=" * 60)

    # 初始化系统
    print("\n🛠️ 初始化RAG系统...")
    conn = Neo4jConnection()
    if not conn.connect():
        print("❌ 数据库连接失败")
        return

    rag = RAGSystem(conn)
    print("✅ 系统初始化完成")

    # 用户会话
    print("\n" + "-" * 60)
    print("开始RAG会话 (输入 'quit' 退出)")
    print("-" * 60)

    while True:
        # 获取用户输入
        user_input = input("\n请输入您的问题: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', '退出']:
            print("👋 会话结束")
            break

        print(f"\n📝 您的问题: {user_input}")

        # 处理查询（自动从kg_schema.json读取实体类型和关系类型）
        result = rag.process_query(
            user_input,
            depth=2,
            similarity_threshold=0.7,
            top_k=5
        )

        # 显示答案
        print("\n💡 系统回答:")
        print(result['answer'])

        # 显示处理信息
        print("\n📊 处理信息:")
        print(f"  相似实体数: {result['similar_entities_count']}")
        print(f"  检索关系数: {len(result['kg_results'])}")
        print(f"  处理时间: {result['processing_time']:.2f}s")

    rag.close()
    conn.close()


if __name__ == "__main__":
    main()
