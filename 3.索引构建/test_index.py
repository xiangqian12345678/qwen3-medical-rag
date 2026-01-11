"""测试索引构建功能"""
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_config_loading():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    from config import ConfigLoader

    loader = ConfigLoader("index.yaml")
    config = loader.config

    print(f"Collection Name: {config.milvus.collection_name}")
    print(f"Auto ID: {config.milvus.auto_id}")

    print("\n稠密向量字段:")
    for name, field_config in config.dense_fields.items():
        if field_config.embed:
            print(f"  - {name}: {field_config.index_field} (type={field_config.type}, dim={field_config.dimension})")

    print("\n稀疏向量字段:")
    for name, field_config in config.sparse_fields.items():
        if field_config.embed:
            print(f"  - {name}: {field_config.index_field}")


def test_knowledge_base():
    """测试知识库初始化和文档添加"""
    print("\n=== 测试知识库 ===")
    from knowledge_base import KnowledgeBase

    # 创建知识库
    kb = KnowledgeBase("index.yaml")

    # 初始化（创建 collection 和索引）
    kb.initialize()

    # 创建测试文档
    from langchain_core.documents import Document
    import hashlib

    documents = [
        Document(
            page_content="高血压是一种常见的慢性疾病，需要长期管理。高血压患者应该定期监测血压，按时服药。",
            metadata={
                "chunk": "高血压是一种常见的慢性疾病，需要长期管理。高血压患者应该定期监测血压，按时服药。",
                "parent_chunk": "高血压管理指南：高血压是一种常见的慢性疾病，需要长期管理。患者应该遵循医生的建议，定期检查血压。",
                "summary": "高血压是一种常见的慢性疾病，需要长期管理和定期监测。",
                "questions": ["什么是高血压？", "如何管理高血压？", "高血压需要长期管理吗？"],
                "document": "高血压管理指南",
                "source": "医学手册",
                "source_name": "manual",
                "lt_doc_id": "doc_001",
                "chunk_id": 0,
                "hash_id": hashlib.md5("高血压".encode()).hexdigest()
            }
        ),
        Document(
            page_content="糖尿病是一种代谢性疾病，主要特征是高血糖。糖尿病患者需要控制饮食，适量运动。",
            metadata={
                "chunk": "糖尿病是一种代谢性疾病，主要特征是高血糖。糖尿病患者需要控制饮食，适量运动。",
                "parent_chunk": "糖尿病管理指南：糖尿病是一种代谢性疾病，主要特征是高血糖。患者需要控制饮食，适量运动。",
                "summary": "糖尿病是一种代谢性疾病，需要控制饮食和适量运动。",
                "questions": ["什么是糖尿病？", "如何控制血糖？"],
                "document": "糖尿病管理指南",
                "source": "医学手册",
                "source_name": "manual",
                "lt_doc_id": "doc_002",
                "chunk_id": 1,
                "hash_id": hashlib.md5("糖尿病".encode()).hexdigest()
            }
        )
    ]

    # 添加文档
    rows = kb.add_documents(documents, show_progress=True)
    print(f"\n添加了 {rows} 行数据")

    return kb


def test_search(kb):
    """测试检索功能"""
    print("\n=== 测试检索 ===")

    # 混合检索
    print("\n1. 混合检索（所有字段）:")
    results = kb.search("什么是高血压？", limit=5, deduplicate=True)
    print(f"返回 {len(results)} 条结果")
    for i, result in enumerate(results):
        entity = result.get("entity", {})
        score = result.get("distance", 0)
        print(f"  [{i+1}] score={score:.4f}, chunk={entity.get('chunk', '')[:50]}...")

    # 单路检索 - questions 字段
    print("\n2. 单路检索（questions_dense）:")
    results = kb.simple_search("什么是高血压？", anns_field="questions_dense", limit=5)
    print(f"返回 {len(results)} 条结果")
    for i, result in enumerate(results):
        entity = result.get("entity", {})
        score = result.get("distance", 0)
        print(f"  [{i+1}] score={score:.4f}, questions={entity.get('questions', '')}")

    # 单路检索 - chunk 字段
    print("\n3. 单路检索（chunk_dense）:")
    results = kb.simple_search("什么是高血压？", anns_field="chunk_dense", limit=5)
    print(f"返回 {len(results)} 条结果")
    for i, result in enumerate(results):
        entity = result.get("entity", {})
        score = result.get("distance", 0)
        print(f"  [{i+1}] score={score:.4f}, chunk={entity.get('chunk', '')[:50]}...")


def test_vectorizer():
    """测试向量化器"""
    print("\n=== 测试向量化器 ===")
    from vectorizer import DocumentVectorizer
    from config import ConfigLoader

    loader = ConfigLoader("index.yaml")
    config = loader.config
    vectorizer = DocumentVectorizer(config)

    # 创建测试文档
    from langchain_core.documents import Document
    import hashlib

    doc = Document(
        page_content="测试文档内容",
        metadata={
            "chunk": "测试文档内容",
            "parent_chunk": "父级测试文档",
            "summary": "测试摘要",
            "questions": ["问题1", "问题2", "问题3"],
            "document": "完整文档",
            "source": "测试",
            "source_name": "test",
            "lt_doc_id": "test_doc",
            "chunk_id": 0,
            "hash_id": hashlib.md5("test".encode()).hexdigest()
        }
    )

    # 向量化文档
    rows = vectorizer.vectorize_document(doc)

    print(f"生成了 {len(rows)} 行数据（包含 list 字段展开）:")
    for i, row in enumerate(rows):
        print(f"  行 {i+1}: pk={row.get('pk')}, vector_id={row.get('vector_id')}")
        print(f"    chunk_dense 维度: {len(row.get('chunk_dense', []))}")
        print(f"    questions_dense 维度: {len(row.get('questions_dense', []))}")
        print(f"    questions: {row.get('questions')}")


if __name__ == "__main__":
    # 测试配置加载
    test_config_loading()

    # 测试向量化器
    test_vectorizer()

    # 测试知识库（需要 Milvus 服务运行）
    try:
        kb = test_knowledge_base()
        test_search(kb)
    except Exception as e:
        print(f"\n知识库测试失败（请确保 Milvus 服务正在运行）: {e}")
