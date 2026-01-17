"""
基础功能测试
测试各个模块的基本功能
"""
from neo4j_connection import Neo4jConnection
from embedding_service import EmbeddingService
from llm_service import LLMService
from config import config, kg_schema


def test_connection():
    """测试数据库连接"""
    print("\n" + "=" * 60)
    print("测试: 数据库连接")
    print("=" * 60)

    try:
        conn = Neo4jConnection()
        if conn.connect():
            print("✅ 数据库连接成功")
            conn.close()
            return True
        else:
            print("❌ 数据库连接失败")
            return False
    except Exception as e:
        print(f"❌ 连接测试异常: {e}")
        return False


def test_embedding():
    """测试嵌入服务"""
    print("\n" + "=" * 60)
    print("测试: 嵌入服务")
    print("=" * 60)

    try:
        service = EmbeddingService()
        text = "测试文本"
        embedding = service.generate_embedding(text)

        if embedding and len(embedding) > 0:
            print(f"✅ 嵌入服务正常")
            print(f"  嵌入维度: {len(embedding)}")
            service.close()
            return True
        else:
            print("❌ 嵌入服务失败")
            service.close()
            return False
    except Exception as e:
        print(f"❌ 嵌入服务异常: {e}")
        return False


def test_llm():
    """测试大模型服务"""
    print("\n" + "=" * 60)
    print("测试: 大模型服务")
    print("=" * 60)

    try:
        service = LLMService()

        # 测试实体关系提取（自动从kg_schema.json读取）
        text = "阿司匹林可以治疗头痛和发热"
        result = service.extract_entities_relations(text)

        if result and ("entities" in result or "relationships" in result):
            print(f"✅ 大模型服务正常")
            print(f"  提取到 {len(result.get('entities', []))} 个实体")
            print(f"  提取到 {len(result.get('relationships', []))} 个关系")
            service.close()
            return True
        else:
            print("❌ 大模型服务失败")
            service.close()
            return False
    except Exception as e:
        print(f"❌ 大模型服务异常: {e}")
        return False


def test_config():
    """测试配置加载"""
    print("\n" + "=" * 60)
    print("测试: 配置加载")
    print("=" * 60)

    try:
        print(f"✅ 配置加载成功")
        print(f"  Neo4j URI: {config.NEO4J_URI}")
        print(f"  Neo4j User: {config.NEO4J_USER}")
        print(f"  嵌入模型: {config.EMBEDDING_MODEL}")
        print(f"\n✅ 知识图谱Schema加载成功")
        print(f"  图谱名称: {kg_schema.schema.get('name', 'N/A')}")
        print(f"  实体类型: {', '.join(kg_schema.get_entity_types())}")
        print(f"  关系类型: {', '.join(kg_schema.get_relationship_types())}")
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行所有基础功能测试")
    print("=" * 60)

    results = {
        "配置加载": test_config(),
        "数据库连接": test_connection(),
        "嵌入服务": test_embedding(),
        "大模型服务": test_llm()
    }

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 测试通过")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
