"""
主程序入口
提供完整的基于知识图谱的RAG单轮会话示例
"""
from embedding_service import EmbeddingService
from llm_service import LLMService
from neo4j_connection import Neo4jConnection
from neo4j_operations import Neo4jOperations
from neo4j_query import Neo4jQuery
from neo4j_save import Neo4jSave
from rag_system import RAGSystem
from text_processor import TextProcessor


def example_1_basic_operations():
    """
    示例1: 基础数据库操作
    演示如何创建实体和关系
    """
    print("\n" + "=" * 60)
    print("示例1: 基础数据库操作")
    print("=" * 60)

    # 连接数据库
    conn = Neo4jConnection()
    if not conn.connect():
        print("❌ 数据库连接失败")
        return

    # 创建操作对象
    ops = Neo4jOperations(conn)

    # 创建实体
    print("\n📝 创建实体...")
    aspirin_id = ops.create_entity(
        name="阿司匹林",
        entity_type="药物",
        properties={"成分": "乙酰水杨酸", "剂量": "100mg"}
    )
    print(f"  创建药物: 阿司匹林, ID: {aspirin_id}")

    headache_id = ops.create_entity("头痛", "症状", {"描述": "头部疼痛"})
    print(f"  创建症状: 头痛, ID: {headache_id}")

    fever_id = ops.create_entity("发热", "症状", {"描述": "体温升高"})
    print(f"  创建症状: 发热, ID: {fever_id}")

    # 创建关系
    print("\n📝 创建关系...")
    rel_id1 = ops.create_relationship(
        source_id=aspirin_id,
        target_id=headache_id,
        rel_type="治疗",
        properties={}
    )
    print(f"  创建关系: 阿司匹林 治疗 头痛, ID: {rel_id1}")

    rel_id2 = ops.create_relationship(
        source_id=aspirin_id,
        target_id=fever_id,
        rel_type="治疗",
        properties={}
    )
    print(f"  创建关系: 阿司匹林 治疗 发热, ID: {rel_id2}")

    # 查询实体
    print("\n📝 查询实体...")
    entity = ops.get_entity_by_id(aspirin_id)
    print(f"  实体详情: {entity}")

    # 获取统计信息
    print("\n📝 获取统计信息...")
    stats = ops.get_statistics()
    print(f"  实体总数: {stats['entities']}")
    print(f"  关系总数: {stats['relationships']}")

    conn.close()


def example_2_extract_and_save():
    """
    示例2: 从文本提取并保存知识
    演示使用大模型从文本提取实体关系并保存到数据库
    """
    print("\n" + "=" * 60)
    print("示例2: 从文本提取并保存知识")
    print("=" * 60)

    # 初始化服务
    conn = Neo4jConnection()
    if not conn.connect():
        print("❌ 数据库连接失败")
        return

    embed_service = EmbeddingService()
    llm_service = LLMService()

    saver = Neo4jSave(conn, embed_service)

    # 示例文本
    text = """
    阿司匹林是一种非甾体抗炎药，常用于治疗头痛、关节痛和发热。
    布洛芬也是一种常用的非甾体抗炎药，主要用于缓解轻中度疼痛和发热。
    感冒会导致头痛、发热、流鼻涕等症状。
    """

    print(f"\n📝 处理文本: {text[:50]}...")

    # 提取并保存知识（自动从kg_schema.json读取实体类型和关系类型）
    success = saver.save_text_knowledge(
        text,
        llm_service
    )

    if success:
        print("✅ 知识保存成功")
    else:
        print("❌ 知识保存失败")

    embed_service.close()
    llm_service.close()
    conn.close()


def example_3_query_knowledge():
    """
    示例3: 查询知识图谱
    演示如何查询知识图谱
    """
    print("\n" + "=" * 60)
    print("示例3: 查询知识图谱")
    print("=" * 60)

    # 连接数据库
    conn = Neo4jConnection()
    if not conn.connect():
        print("❌ 数据库连接失败")
        return

    query = Neo4jQuery(conn)

    # 关键字搜索
    print("\n📝 关键字搜索...")
    result = query.search_by_keyword("阿司匹林", limit=20)
    print(f"  找到 {len(result['nodes'])} 个节点")
    print(f"  找到 {len(result['links'])} 条关系")

    # 三元组查询
    print("\n📝 三元组查询...")
    result = query.query_triples(head="阿司匹林", relation="治疗")
    print(f"  找到 {len(result['links'])} 条匹配关系")

    # 获取整个图谱
    print("\n📝 获取整个图谱...")
    graph = query.get_all_graph(limit=50)
    print(f"  图谱包含 {len(graph['nodes'])} 个节点")
    print(f"  图谱包含 {len(graph['links'])} 条关系")

    conn.close()


def example_4_rag_query():
    """
    示例4: RAG查询
    演示完整的RAG查询流程
    """
    print("\n" + "=" * 60)
    print("示例4: RAG查询")
    print("=" * 60)

    # 初始化RAG系统
    conn = Neo4jConnection()
    if not conn.connect():
        print("❌ 数据库连接失败")
        return

    rag = RAGSystem(conn)

    # 查询示例
    queries = [
        "阿司匹林可以治疗什么？",
        "头痛有什么症状？",
        "感冒会导致什么症状？"
    ]

    print("\n📝 处理用户查询...")

    for query_text in queries:
        print(f"\n问题: {query_text}")
        result = rag.process_query(
            query_text,
            depth=2,
            similarity_threshold=0.7,
            top_k=5
        )

        print(f"答案: {result['answer']}")
        print(f"处理时间: {result['processing_time']:.2f}s")

    rag.close()
    conn.close()


def example_5_process_documents():
    """
    示例5: 处理文档并构建知识库
    演示从文档文件批量处理并构建知识库
    """
    print("\n" + "=" * 60)
    print("示例5: 处理文档并构建知识库")
    print("=" * 60)

    # 初始化服务
    conn = Neo4jConnection()
    if not conn.connect():
        print("❌ 数据库连接失败")
        return

    embed_service = EmbeddingService()
    llm_service = LLMService()
    saver = Neo4jSave(conn, embed_service)
    processor = TextProcessor()

    # 处理文档目录
    documents_dir = "../data/graph"

    if not os.path.exists(documents_dir):
        print(f"⚠️ 文档目录不存在: {documents_dir}")
        print("  请确保在tmp目录下创建documents目录并放入文档文件")
        return

    print(f"\n📝 处理文档目录: {documents_dir}")

    # 加载所有文本
    texts = processor.load_text_from_directory(documents_dir)

    if not texts:
        print("⚠️ 未找到可处理的文本")
        return

    print(f"共加载 {len(texts)} 段文本")

    # 分割并处理每段文本（自动从kg_schema.json读取实体类型和关系类型）
    for i, text in enumerate(texts[:10]):  # 限制处理前10段文本
        print(f"\n处理文本 {i + 1}/{len(texts)}...")
        print(f"文本内容: {text[:100]}...")

        saver.save_text_knowledge(
            text,
            llm_service
        )

    print("\n✅ 文档处理完成")

    # 获取统计信息
    ops = Neo4jOperations(conn)
    stats = ops.get_statistics()
    print(f"\n知识库统计:")
    print(f"  实体总数: {stats['entities']}")
    print(f"  关系总数: {stats['relationships']}")

    embed_service.close()
    llm_service.close()
    conn.close()


def example_6_complete_rag_session():
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


def main():
    """
    主函数
    运行所有示例
    """
    print("=" * 60)
    print("基于知识图谱的RAG系统 - 完整示例")
    print("=" * 60)

    # 运行示例
    example_1_basic_operations()
    example_2_extract_and_save()
    example_3_query_knowledge()
    example_4_rag_query()
    example_5_process_documents()
    example_6_complete_rag_session()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 导入os模块
    import os

    # 检查documents目录
    if not os.path.exists("../data/documents"):
        os.makedirs("../data/documents", exist_ok=True)
        print(" 已创建documents目录")
        print("  请将文档文件放入此目录后再运行示例5和示例6")

    # 运行主函数
    main()
