"""
简单测试知识图谱检索
"""
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from kg_loader import KGraphConfigLoader
from neo4j_connection import Neo4jConnection
from graph_searcher import GraphSearcher


def main():
    print("=" * 60)
    print("知识图谱检索测试")
    print("=" * 60)

    try:
        # 加载配置
        config = KGraphConfigLoader()

        print(f"\n📊 配置信息:")
        print(f"   Neo4j URI: {config.neo4j_config.uri}")
        print(f"   数据库: {config.neo4j_config.database}")

        # 创建连接
        print(f"\n🔌 连接Neo4j...")
        neo4j_conn = Neo4jConnection(config)
        connected = neo4j_conn.connect()

        if not connected:
            print(f"❌ 连接失败")
            return

        print(f"✅ 连接成功")

        # 创建嵌入配置
        embedding_config = {
            "provider": config.get("embedding.provider", "ollama"),
            "model": config.get("embedding.model", "nomic-embed-text"),
            "api_key": config.get("embedding.api_key", None),
            "base_url": config.get("embedding.base_url", "http://localhost:11434/v1")
        }

        print(f"\n📝 嵌入配置:")
        print(f"   Provider: {embedding_config['provider']}")
        print(f"   Model: {embedding_config['model']}")

        # 创建图谱检索器
        print(f"\n🔍 创建图谱检索器...")
        graph_searcher = GraphSearcher(neo4j_conn, embedding_config=embedding_config)
        print(f"✅ 图谱检索器创建成功")

        # 测试1: 关键词检索
        print(f"\n" + "=" * 60)
        print("测试1: 关键词检索")
        print("=" * 60)
        keyword = "阿司匹林"
        print(f"搜索关键词: '{keyword}'")
        docs = graph_searcher.search_by_keyword(keyword, limit=5)
        print(f"✅ 找到 {len(docs)} 个实体:")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc.page_content}")

        # 测试2: 关系检索
        print(f"\n" + "=" * 60)
        print("测试2: 关系检索")
        print("=" * 60)
        entity_name = "阿司匹林"
        print(f"查询实体: '{entity_name}' 的关系")
        docs = graph_searcher.search_by_relation(entity_name, limit=5)
        print(f"✅ 找到 {len(docs)} 条关系:")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc.page_content}")

        # 测试3: 向量检索
        print(f"\n" + "=" * 60)
        print("测试3: 向量检索")
        print("=" * 60)
        query = "糖尿病"
        print(f"查询: '{query}'")
        docs = graph_searcher.search_by_vector(query, threshold=0.5, top_k=5)
        print(f"✅ 找到 {len(docs)} 个相似实体:")
        for i, doc in enumerate(docs, 1):
            similarity = doc.metadata.get("similarity", 0)
            print(f"   {i}. {doc.page_content} (相似度: {similarity:.3f})")

        # 测试4: 综合检索
        print(f"\n" + "=" * 60)
        print("测试4: 综合检索")
        print("=" * 60)
        query = "高血压"
        print(f"查询: '{query}'")
        result = graph_searcher.search_graph_by_query(query, top_k=5, similarity_threshold=0.5)
        vdb_results = result.get("vdb_results", [])
        print(f"✅ 找到 {len(vdb_results)} 条结果:")
        for i, doc in enumerate(vdb_results, 1):
            print(f"   {i}. {doc}")

        # 关闭连接
        neo4j_conn.close()
        print(f"\n✅ 连接已关闭")

        print(f"\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
