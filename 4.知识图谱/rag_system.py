"""
RAG系统模块
整合向量检索和知识图谱，生成最终答案
"""
from typing import Dict, List
from neo4j_connection import Neo4jConnection
from neo4j_query import Neo4jQuery
from vector_search import VectorSearch
from embedding_service import EmbeddingService
from llm_service import LLMService
from config import kg_schema


class RAGSystem:
    """
    RAG系统类
    整合向量检索和知识图谱检索
    """

    def __init__(self, connection: Neo4jConnection = None):
        """
        初始化RAG系统

        【输入示例】
        conn = Neo4jConnection()
        conn.connect()
        rag = RAGSystem(conn)

        【输出示例】
        None (系统已初始化)
        """
        self.connection = connection or Neo4jConnection()

        # 初始化各个组件
        self.embed_service = EmbeddingService()
        self.llm_service = LLMService()
        self.query = Neo4jQuery(self.connection)
        self.search = VectorSearch(self.connection, self.embed_service)

        # 加载嵌入向量
        self.search.load_embeddings_from_db()

    def process_query(self, query_text: str,
                      entity_types: List[str] = None,
                      relation_types: List[str] = None,
                      depth: int = 2,
                      similarity_threshold: float = 0.7,
                      top_k: int = 5) -> Dict:
        """
        处理用户查询

        【输入示例】
        result = rag.process_query(
            query_text="阿司匹林可以治疗什么？",
            entity_types=["药物", "症状", "疾病"],
            relation_types=["治疗", "导致"],
            depth=2,
            similarity_threshold=0.75,
            top_k=5
        )

        【输出示例】
        {
            "answer": "根据知识图谱，阿司匹林主要用于治疗头痛、发热等症状...",
            "kg_results": [...],
            "similar_entities": [...],
            "processing_time": 2.34
        }
        """
        import time
        start_time = time.time()

        print(f"\n🔍 处理查询: {query_text}")

        # 1. 提取查询中的实体
        # 如果没有指定实体类型和关系类型，从kg_schema.json读取
        entity_types = entity_types or kg_schema.get_entity_types()
        relation_types = relation_types or kg_schema.get_relationship_types()

        extraction_result = self.llm_service.extract_entities_relations(
            query_text, entity_types, relation_types
        )

        entities = extraction_result.get("entities", [])

        # 2. 向量检索相似实体
        entity_texts = [
            f"{e.get('type', '实体')}: {e['name']}"
            for e in entities
        ]

        all_similar_entity_ids = set()

        for entity_text in entity_texts:
            similar_entities = self.search.search_similar_entities(
                entity_text,
                threshold=similarity_threshold,
                top_k=top_k
            )

            for entity in similar_entities:
                all_similar_entity_ids.add(entity["id"])

        print(f"🔍 找到 {len(all_similar_entity_ids)} 个相似实体")

        # 3. 知识图谱查询
        kg_results = []
        if all_similar_entity_ids:
            kg_results = self.query.query_by_entities(
                list(all_similar_entity_ids),
                depth=depth
            )

        print(f"📊 查询到 {len(kg_results)} 条关系")

        # 4. 生成答案
        vdb_results = [result.get("source", "") for result in kg_results[:5]]
        answer = self.llm_service.generate_rag_answer(
            query_text,
            kg_results,
            vdb_results
        )

        processing_time = time.time() - start_time

        return {
            "answer": answer,
            "kg_results": kg_results[:10],
            "similar_entities_count": len(all_similar_entity_ids),
            "processing_time": processing_time
        }

    def simple_query(self, query_text: str) -> str:
        """
        简单查询接口

        【输入示例】
        answer = rag.simple_query("阿司匹林有什么作用？")

        【输出示例】
        "根据知识图谱，阿司匹林主要用于治疗头痛、发热等症状..."
        """
        result = self.process_query(query_text)
        return result["answer"]

    def get_graph_data(self, limit: int = 100) -> Dict:
        """
        获取图谱数据用于可视化

        【输入示例】
        data = rag.get_graph_data(limit=50)

        【输出示例】
        {
            "nodes": [...],
            "links": [...]
        }
        """
        return self.query.get_all_graph(limit=limit)

    def search_graph(self, keyword: str) -> Dict:
        """
        搜索图谱

        【输入示例】
        data = rag.search_graph("阿司匹林")

        【输出示例】
        {
            "nodes": [...],
            "links": [...]
        }
        """
        return self.query.search_by_keyword(keyword)

    def close(self):
        """
        关闭系统

        【输入示例】
        rag.close()

        【输出示例】
        None (系统已关闭)
        """
        self.embed_service.close()
        self.llm_service.close()


# 使用示例
if __name__ == "__main__":
    # 示例1: 初始化系统
    print("示例1: 初始化RAG系统")
    conn = Neo4jConnection()
    conn.connect()

    rag = RAGSystem(conn)

    # 示例2: 处理查询
    print("\n示例2: 处理查询")
    queries = [
        "阿司匹林可以治疗什么？",
        "头痛有什么症状？",
        "感冒会导致什么？"
    ]

    for query in queries:
        print(f"\n问题: {query}")
        result = rag.process_query(query)
        print(f"答案: {result['answer'][:100]}...")
        print(f"处理时间: {result['processing_time']:.2f}s")

    # 示例3: 简单查询
    print("\n示例3: 简单查询")
    answer = rag.simple_query("阿司匹林有什么副作用？")
    print(f"答案: {answer}")

    # 示例4: 获取图谱数据
    print("\n示例4: 获取图谱数据")
    graph_data = rag.get_graph_data(limit=10)
    print(f"图谱包含 {len(graph_data['nodes'])} 个节点")
    print(f"图谱包含 {len(graph_data['links'])} 条关系")

    # 示例5: 搜索图谱
    print("\n示例5: 搜索图谱")
    search_result = rag.search_graph("阿司匹林")
    print(f"找到 {len(search_result['nodes'])} 个相关节点")

    rag.close()
    conn.close()
