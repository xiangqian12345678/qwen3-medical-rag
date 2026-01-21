"""
诊断Neo4j数据库中的数据
"""
import logging
from neo4j_connection import Neo4jConnection
from kg_loader import KGraphConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("Neo4j数据库诊断")
    print("=" * 60)

    try:
        # 加载配置
        config = KGraphConfigLoader()

        print(f"\n📊 配置信息:")
        print(f"   Neo4j URI: {config.neo4j_config.uri}")
        print(f"   数据库: {config.neo4j_config.database}")

        # 创建连接
        print(f"\n🔌 连接Neo4j...")
        conn = Neo4jConnection(config)
        connected = conn.connect()

        if not connected:
            print(f"❌ 连接失败")
            return

        print(f"✅ 连接成功")

        driver = conn.get_driver()

        with driver.session(database=config.neo4j_config.database) as session:
            # 1. 检查总节点数
            print(f"\n" + "=" * 60)
            print("1. 检查节点总数")
            print("=" * 60)
            result = session.run("MATCH (n) RETURN count(n) as count")
            total_nodes = result.single()["count"]
            print(f"✅ 总节点数: {total_nodes}")

            # 2. 检查节点类型分布
            print(f"\n" + "=" * 60)
            print("2. 节点类型分布")
            print("=" * 60)
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC LIMIT 10"
            )
            for record in result:
                print(f"   {record['type']:20s}: {record['count']} 个")

            # 3. 检查关系总数
            print(f"\n" + "=" * 60)
            print("3. 检查关系总数")
            print("=" * 60)
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            total_rels = result.single()["count"]
            print(f"✅ 总关系数: {total_rels}")

            # 4. 检查关系类型分布
            print(f"\n" + "=" * 60)
            print("4. 关系类型分布")
            print("=" * 60)
            result = session.run(
                "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC LIMIT 10"
            )
            for record in result:
                print(f"   {record['type']:20s}: {record['count']} 条")

            # 5. 检查嵌入向量数量
            print(f"\n" + "=" * 60)
            print("5. 检查嵌入向量")
            print("=" * 60)
            result = session.run(
                "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) as count"
            )
            embedding_count = result.single()["count"]
            print(f"✅ 有嵌入向量的节点数: {embedding_count}")

            # 6. 查看样本节点
            print(f"\n" + "=" * 60)
            print("6. 查看样本节点（前10个）")
            print("=" * 60)
            result = session.run(
                "MATCH (n) RETURN n.name as name, labels(n)[0] as type LIMIT 10"
            )
            for record in result:
                print(f"   {record['type']:15s}: {record['name']}")

            # 7. 搜索特定关键词
            print(f"\n" + "=" * 60)
            print("7. 搜索特定关键词")
            print("=" * 60)
            keywords = ["糖尿病", "高血压", "房颤", "阿司匹林"]
            for kw in keywords:
                result = session.run(
                    "MATCH (n) WHERE n.name CONTAINS $kw RETURN count(n) as count", kw=kw
                )
                count = result.single()["count"]
                print(f"   '{kw}': 找到 {count} 个节点")

            # 8. 查看阿司匹林的关系
            print(f"\n" + "=" * 60)
            print("8. 查看'阿司匹林'的关系")
            print("=" * 60)
            result = session.run(
                "MATCH (a {name:'阿司匹林'})-[r]->(b) RETURN type(r) as relation, b.name as target LIMIT 5"
            )
            relations = list(result)
            if relations:
                for rel in relations:
                    print(f"   阿司匹林 -> {rel['relation']} -> {rel['target']}")
            else:
                print("   未找到关系")

        conn.close()
        print(f"\n✅ 连接已关闭")

    except Exception as e:
        print(f"❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
