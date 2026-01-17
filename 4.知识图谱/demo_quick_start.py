"""
快速开始示例
展示如何使用RAG系统进行问答
"""
from neo4j_connection import Neo4jConnection
from rag_system import RAGSystem
from config import kg_schema


def quick_start_demo():
    """
    快速开始演示
    展示完整的RAG问答流程
    """
    print("=" * 70)
    print("基于知识图谱的RAG系统 - 快速开始")
    print("=" * 70)

    # 显示知识图谱配置
    print("\n[知识图谱配置]")
    print(f"  图谱名称: {kg_schema.schema.get('name', 'N/A')}")
    print(f"  实体类型: {', '.join(kg_schema.get_entity_types())}")
    print(f"  关系类型: {', '.join(kg_schema.get_relationship_types())}")

    # 步骤1: 连接数据库
    print("\n[步骤1] 连接Neo4j数据库...")
    conn = Neo4jConnection()
    if not conn.connect():
        print("❌ 数据库连接失败，请检查配置")
        print("  - 确保Neo4j已启动")
        print("  - 检查config/config.json中的连接信息")
        return

    print("✅ 数据库连接成功")

    # 步骤2: 初始化RAG系统
    print("\n[步骤2] 初始化RAG系统...")
    rag = RAGSystem(conn)
    print("✅ RAG系统初始化完成")

    # 步骤3: 演示查询
    print("\n[步骤3] 演示查询功能...")

    demo_queries = [
        "阿司匹林可以治疗什么？",
        "头痛有哪些症状？",
        "感冒会导致什么？"
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*70}")
        print(f"查询 {i}: {query}")
        print(f"{'='*70}")

        # 处理查询（自动从kg_schema.json读取实体类型和关系类型）
        result = rag.process_query(
            query,
            depth=2,
            similarity_threshold=0.7,
            top_k=5
        )

        # 显示结果
        print(f"\n💡 系统回答:")
        print(result['answer'])

        print(f"\n📊 查询详情:")
        print(f"  - 相似实体数: {result['similar_entities_count']}")
        print(f"  - 检索关系数: {len(result['kg_results'])}")
        print(f"  - 处理时间: {result['processing_time']:.2f}秒")

    # 步骤4: 交互式查询
    print(f"\n{'='*70}")
    print("[步骤4] 交互式查询")
    print(f"{'='*70}")
    print("提示: 输入您的问题，输入 'quit' 退出\n")

    while True:
        try:
            user_input = input("👤 您的问题: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 感谢使用！")
                break

            # 处理查询
            result = rag.simple_query(user_input)

            # 显示答案
            print(f"\n🤖 系统回答:")
            print(result)

        except KeyboardInterrupt:
            print("\n\n👋 感谢使用！")
            break
        except Exception as e:
            print(f"\n❌ 处理查询时出错: {e}")

    # 关闭连接
    print("\n[步骤5] 关闭系统...")
    rag.close()
    conn.close()
    print("✅ 系统已关闭")


if __name__ == "__main__":
    quick_start_demo()
