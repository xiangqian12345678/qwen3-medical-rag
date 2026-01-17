"""
向量检索模块
提供基于嵌入向量的相似度检索功能
"""
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from neo4j_connection import Neo4jConnection
from embedding_service import EmbeddingService


class VectorSearch:
    """
    向量检索类
    基于嵌入向量进行相似度检索
    """

    def __init__(self, connection: Neo4jConnection = None,
                 embedding_service: EmbeddingService = None):
        """
        初始化向量检索对象

        【输入示例】
        conn = Neo4jConnection()
        conn.connect()
        embed_service = EmbeddingService()
        search = VectorSearch(conn, embed_service)

        【输出示例】
        None (检索对象已初始化)
        """
        self.connection = connection or Neo4jConnection()
        if self.connection.check_connection():
            self.driver = self.connection.get_driver()
        else:
            self.driver = None

        self.embedding_service = embedding_service or EmbeddingService()

        # 向量索引
        self.entity_index = {
            "ids": [],
            "names": [],
            "types": [],
            "embeddings": np.empty((0, 1536))
        }

        self.rel_index = {
            "ids": [],
            "types": [],
            "sources": [],
            "targets": [],
            "embeddings": np.empty((0, 1536))
        }

        # ANN模型
        self.entity_ann = None
        self.rel_ann = None

        # 参数
        self.similarity_threshold = 0.7
        self.top_k = 5

    def load_embeddings_from_db(self) -> bool:
        """
        从数据库加载所有嵌入向量

        【输入示例】
        success = search.load_embeddings_from_db()

        【输出示例】
        ✅ 从数据库加载了 150 个实体和 320 个关系
        返回: True
        """
        if not self.driver:
            print("❌ 数据库未连接")
            return False

        try:
            with self.driver.session() as session:
                # 加载实体嵌入
                print("🔄 加载实体嵌入...")
                result = session.run("""
                    MATCH (e)
                    WHERE e.embedding IS NOT NULL
                    RETURN elementId(e) as id, e.name as name,
                           labels(e)[0] as type, e.embedding as embedding
                """)

                entities = []
                for record in result:
                    if record["embedding"]:
                        entities.append({
                            "id": record["id"],
                            "name": record["name"],
                            "type": record["type"],
                            "embedding": record["embedding"]
                        })

                # 更新实体索引
                self.entity_index = {
                    "ids": [],
                    "names": [],
                    "types": [],
                    "embeddings": np.empty((0, 1536))
                }

                for entity in entities:
                    self.entity_index["ids"].append(str(entity["id"]))
                    self.entity_index["names"].append(entity["name"])
                    self.entity_index["types"].append(entity["type"])
                    self.entity_index["embeddings"] = np.vstack([
                        self.entity_index["embeddings"],
                        np.array(entity["embedding"]).reshape(1, -1)
                    ])

                print(f"✅ 加载了 {len(self.entity_index['ids'])} 个实体嵌入")

                # 加载关系嵌入
                print("🔄 加载关系嵌入...")
                result = session.run("""
                    MATCH ()-[r]->()
                    WHERE r.embedding IS NOT NULL
                    RETURN elementId(r) as id, type(r) as type,
                           startNode(r).name as source,
                           endNode(r).name as target,
                           r.embedding as embedding
                """)

                relationships = []
                for record in result:
                    if record["embedding"]:
                        relationships.append({
                            "id": record["id"],
                            "type": record["type"],
                            "source": record["source"],
                            "target": record["target"],
                            "embedding": record["embedding"]
                        })

                # 更新关系索引
                self.rel_index = {
                    "ids": [],
                    "types": [],
                    "sources": [],
                    "targets": [],
                    "embeddings": np.empty((0, 1536))
                }

                for rel in relationships:
                    self.rel_index["ids"].append(rel["id"])
                    self.rel_index["types"].append(rel["type"])
                    self.rel_index["sources"].append(rel["source"])
                    self.rel_index["targets"].append(rel["target"])
                    self.rel_index["embeddings"] = np.vstack([
                        self.rel_index["embeddings"],
                        np.array(rel["embedding"]).reshape(1, -1)
                    ])

                print(f"✅ 加载了 {len(self.rel_index['ids'])} 个关系嵌入")

                # 构建ANN模型
                self._build_ann_models()

                return True
        except Exception as e:
            print(f"❌ 加载嵌入失败: {e}")
            return False

    def _build_ann_models(self):
        """
        构建ANN模型
        """
        # 构建实体ANN模型
        if self.entity_index["embeddings"].shape[0] > 0:
            max_k = max(1, self.entity_index["embeddings"].shape[0] - 1)
            k = min(self.top_k * 2, max_k)
            self.entity_ann = NearestNeighbors(n_neighbors=k, metric='cosine')
            self.entity_ann.fit(self.entity_index["embeddings"])

        # 构建关系ANN模型
        if self.rel_index["embeddings"].shape[0] > 0:
            max_k = max(1, self.rel_index["embeddings"].shape[0] - 1)
            k = min(self.top_k * 2, max_k)
            self.rel_ann = NearestNeighbors(n_neighbors=k, metric='cosine')
            self.rel_ann.fit(self.rel_index["embeddings"])

    def search_similar_entities(self, query_text: str,
                                threshold: float = None,
                                top_k: int = None) -> List[Dict]:
        """
        搜索相似实体

        【输入示例】
        results = search.search_similar_entities(
            query_text="药物: 阿司匹林",
            threshold=0.75,
            top_k=5
        )

        【输出示例】
        [
            {"id": "123", "name": "阿司匹林", "type": "药物", "similarity": 0.95},
            {"id": "456", "name": "布洛芬", "type": "药物", "similarity": 0.88}
        ]
        """
        threshold = threshold or self.similarity_threshold
        top_k = top_k or self.top_k

        # 生成查询嵌入
        query_embedding = self.embedding_service.generate_embedding(query_text)
        if not query_embedding:
            return []

        query_vector = np.array(query_embedding).reshape(1, -1)

        # 使用ANN模型检索
        if self.entity_ann:
            distances, indices = self.entity_ann.kneighbors(query_vector)

            similar_entities = []
            for idx, dist in zip(indices[0], distances[0]):
                similarity = 1 - dist

                if similarity >= threshold:
                    similar_entities.append({
                        "id": self.entity_index["ids"][idx],
                        "name": self.entity_index["names"][idx],
                        "type": self.entity_index["types"][idx],
                        "similarity": similarity
                    })

            # 排序并截取top_k
            similar_entities.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_entities[:top_k]

        # 暴力搜索
        similarities = cosine_similarity(query_vector, self.entity_index["embeddings"])[0]

        similar_entities = []
        for idx, sim in enumerate(similarities):
            if sim >= threshold:
                similar_entities.append({
                    "id": self.entity_index["ids"][idx],
                    "name": self.entity_index["names"][idx],
                    "type": self.entity_index["types"][idx],
                    "similarity": sim
                })

        similar_entities.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_entities[:top_k]

    def search_similar_relationships(self, query_text: str,
                                       threshold: float = None,
                                       top_k: int = None) -> List[Dict]:
        """
        搜索相似关系

        【输入示例】
        results = search.search_similar_relationships(
            query_text="治疗: 阿司匹林 -> 头痛",
            threshold=0.7,
            top_k=5
        )

        【输出示例】
        [
            {
                "id": "789",
                "type": "治疗",
                "source": "阿司匹林",
                "target": "头痛",
                "similarity": 0.92
            }
        ]
        """
        threshold = threshold or self.similarity_threshold
        top_k = top_k or self.top_k

        # 生成查询嵌入
        query_embedding = self.embedding_service.generate_embedding(query_text)
        if not query_embedding:
            return []

        query_vector = np.array(query_embedding).reshape(1, -1)

        # 使用ANN模型检索
        if self.rel_ann:
            distances, indices = self.rel_ann.kneighbors(query_vector)

            similar_rels = []
            for idx, dist in zip(indices[0], distances[0]):
                similarity = 1 - dist

                if similarity >= threshold:
                    similar_rels.append({
                        "id": self.rel_index["ids"][idx],
                        "type": self.rel_index["types"][idx],
                        "source": self.rel_index["sources"][idx],
                        "target": self.rel_index["targets"][idx],
                        "similarity": similarity
                    })

            # 排序并截取top_k
            similar_rels.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_rels[:top_k]

        # 暴力搜索
        similarities = cosine_similarity(query_vector, self.rel_index["embeddings"])[0]

        similar_rels = []
        for idx, sim in enumerate(similarities):
            if sim >= threshold:
                similar_rels.append({
                    "id": self.rel_index["ids"][idx],
                    "type": self.rel_index["types"][idx],
                    "source": self.rel_index["sources"][idx],
                    "target": self.rel_index["targets"][idx],
                    "similarity": sim
                })

        similar_rels.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_rels[:top_k]


# 使用示例
if __name__ == "__main__":
    # 示例1: 加载嵌入向量
    print("示例1: 加载嵌入向量")
    conn = Neo4jConnection()
    conn.connect()

    embed_service = EmbeddingService()
    search = VectorSearch(conn, embed_service)

    search.load_embeddings_from_db()

    # 示例2: 搜索相似实体
    print("\n示例2: 搜索相似实体")
    results = search.search_similar_entities("药物: 阿司匹林", top_k=3)
    print(f"找到 {len(results)} 个相似实体:")
    for r in results:
        print(f"  - {r['name']} ({r['type']}): {r['similarity']:.3f}")

    # 示例3: 搜索相似关系
    print("\n示例3: 搜索相似关系")
    results = search.search_similar_relationships("治疗: 阿司匹林 -> 头痛", top_k=3)
    print(f"找到 {len(results)} 个相似关系:")
    for r in results:
        print(f"  - {r['source']} {r['type']} {r['target']}: {r['similarity']:.3f}")

    embed_service.close()
    conn.close()
