"""
知识图谱构建脚本
处理文档并将提取的知识保存到 Neo4j 数据库
"""
import os
from embedding_service import EmbeddingService
from llm_service import LLMService
from neo4j_connection import Neo4jConnection
from neo4j_operations import Neo4jOperations
from neo4j_save import Neo4jSave
from text_processor import TextProcessor


def process_documents_to_kg(documents_dir: str = "../data/graph"):
    """
    处理文档并构建知识图谱

    Args:
        documents_dir: 文档目录路径
    """
    print("=" * 60)
    print("知识图谱构建 - 文档处理")
    print("=" * 60)

    # 检查文档目录
    if not os.path.exists(documents_dir):
        print(f"❌ 文档目录不存在: {documents_dir}")
        return False

    # 初始化服务
    print("\n🛠️ 初始化服务...")
    conn = Neo4jConnection()
    if not conn.connect():
        print("❌ 数据库连接失败")
        return False
    print("✅ 数据库连接成功")

    embed_service = EmbeddingService()
    llm_service = LLMService()
    saver = Neo4jSave(conn, embed_service)
    processor = TextProcessor()

    # 加载文档
    print(f"\n📂 加载文档: {documents_dir}")
    texts = processor.load_text_from_directory(documents_dir)

    if not texts:
        print("❌ 未找到可处理的文本")
        embed_service.close()
        llm_service.close()
        conn.close()
        return False

    print(f"✅ 共加载 {len(texts)} 段文本")

    # 处理每段文本并保存到 Neo4j
    print("\n🔄 开始提取知识并保存到 Neo4j...")
    success_count = 0
    fail_count = 0

    for i, text in enumerate(texts):
        print(f"\n[{i + 1}/{len(texts)}] 处理中...")
        print(f"文本预览: {text[:80]}...")

        try:
            saver.save_text_knowledge(text, llm_service)
            success_count += 1
            print("✅ 保存成功")
        except Exception as e:
            fail_count += 1
            print(f"❌ 保存失败: {e}")

    # 获取统计信息
    ops = Neo4jOperations(conn)
    stats = ops.get_statistics()

    # 输出结果
    print("\n" + "=" * 60)
    print("知识图谱构建完成")
    print("=" * 60)
    print(f"\n📊 处理统计:")
    print(f"  成功处理: {success_count} 段")
    print(f"  处理失败: {fail_count} 段")
    print(f"\n📚 知识库统计:")
    print(f"  实体总数: {stats['entities']}")
    print(f"  关系总数: {stats['relationships']}")

    # 关闭连接
    embed_service.close()
    llm_service.close()
    conn.close()

    print("\n✅ 所有服务已关闭")
    return True


if __name__ == "__main__":
    # 处理文档并构建知识图谱
    process_documents_to_kg()
