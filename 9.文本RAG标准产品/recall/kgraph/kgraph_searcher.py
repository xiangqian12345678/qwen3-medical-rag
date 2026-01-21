"""
知识图谱检索模块
提供基于Neo4j的图谱查询功能，支持向量相似度检索
"""
import logging
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import numpy as np
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from .kgraph_schema import kg_schema
from .neo4j_connection import Neo4jConnection

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_gs_instance: Optional["GraphSearcher"] = None


class GraphSearcher:
    """
    图谱检索类
    提供基于向量相似度和关键词的图谱检索功能，将查询结果转换为Document对象
    """

    def __init__(self, connection: Neo4jConnection, database: str = None, embedding_config=None):
        """
        初始化图谱检索器

        Args:
            connection: Neo4j连接对象
            database: 数据库名称（可选）
            embedding_config: 嵌入模型配置
        """
        connected = connection.connect()
        if not connected:
            logger.warning(f"Neo4j连接失败: {connection.uri}")

        self.connection = connection
        self.database = database or self.connection.database
        if database:
            self.connection.database = database
        if self.connection.check_connection():
            self.driver = self.connection.get_driver()
        else:
            self.driver = None

        # 初始化向量检索
        self.embedding_config = embedding_config
        self.entity_index = {
            "ids": [],
            "names": [],
            "types": [],
            "embeddings": np.empty((0, 1536))  # 默认维度
        }
        self.embedding_client = None

        # 加载向量索引
        self._load_vector_index()

    def _load_vector_index(self):
        """从数据库加载向量索引"""
        if not self.driver:
            logger.warning("Neo4j驱动未连接，无法加载向量索引")
            return

        if not self.embedding_config:
            logger.warning("未配置嵌入模型，无法加载向量索引")
            return

        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_ollama import OllamaEmbeddings
            from langchain_community.embeddings import DashScopeEmbeddings

            # 创建嵌入客户端
            if self.embedding_config.get("provider") == "openai":
                self.embedding_client = OpenAIEmbeddings(
                    model=self.embedding_config.get("model"),
                    api_key=self.embedding_config.get("api_key"),
                    base_url=self.embedding_config.get("base_url")
                )
            elif self.embedding_config.get("provider") == "ollama":
                self.embedding_client = OllamaEmbeddings(
                    model=self.embedding_config.get("model"),
                    base_url=self.embedding_config.get("base_url")
                )
            elif self.embedding_config.get("provider") == "dashscope":
                self.embedding_client = DashScopeEmbeddings(
                    model=self.embedding_config.get("model"),
                    dashscope_api_key=self.embedding_config.get("api_key")
                )
            else:
                logger.warning(f"不支持的嵌入提供商: {self.embedding_config.get('provider')}")
                return

            with self.driver.session(database=self.database) as session:
                # 加载实体嵌入
                result = session.run(
                    """
                    MATCH (e)
                    WHERE e.embedding IS NOT NULL
                    RETURN elementId(e) as id, e.name as name,
                           labels(e)[0] as type, e.embedding as embedding
                    LIMIT 10000
                    """
                )

                entities = []
                for record in result:
                    if record["embedding"]:
                        entities.append({
                            "id": record["id"],
                            "name": record["name"],
                            "type": record["type"],
                            "embedding": record["embedding"]
                        })

                if entities:
                    # 重置索引
                    self.entity_index = {
                        "ids": [],
                        "names": [],
                        "types": [],
                        "embeddings": np.empty((0, len(entities[0]["embedding"])))
                    }

                    for entity in entities:
                        self.entity_index["ids"].append(str(entity["id"]))
                        self.entity_index["names"].append(entity["name"])
                        self.entity_index["types"].append(entity["type"])
                        self.entity_index["embeddings"] = np.vstack([
                            self.entity_index["embeddings"],
                            np.array(entity["embedding"]).reshape(1, -1)
                        ])

                    logger.info(f"成功加载 {len(self.entity_index['ids'])} 个实体向量索引")
                else:
                    logger.warning("数据库中没有找到实体嵌入向量")

        except Exception as e:
            logger.error(f"加载向量索引失败: {e}")

    def search_by_vector(self, query_text: str, threshold: float = 0.5, top_k: int = 5) -> List[Document]:
        """
        基于向量相似度搜索实体

        Args:
            query_text: 查询文本
            threshold: 相似度阈值
            top_k: 返回结果数量限制

        Returns:
            Document对象列表
        """
        if not self.embedding_client or len(self.entity_index["embeddings"]) == 0:
            logger.warning("向量检索不可用，回退到关键词检索")
            return self.search_by_keyword(query_text, top_k)

        try:
            # 生成查询嵌入
            query_embedding = self.embedding_client.embed_query(query_text)
            query_vector = np.array(query_embedding).reshape(1, -1)

            # 计算相似度
            similarities = cosine_similarity(query_vector, self.entity_index["embeddings"])[0]

            documents = []
            for idx, sim in enumerate(similarities):
                if sim >= threshold:
                    content = f"{self.entity_index['types'][idx]}: {self.entity_index['names'][idx]}"
                    metadata = {
                        "node_id": self.entity_index["ids"][idx],
                        "entity_name": self.entity_index["names"][idx],
                        "entity_type": self.entity_index["types"][idx],
                        "similarity": float(sim),
                        "source": "knowledge_graph"
                    }

                    documents.append(Document(page_content=content, metadata=metadata))

            # 排序并截取top_k
            documents.sort(key=lambda x: x.metadata.get("similarity", 0), reverse=True)
            logger.info(f"向量检索找到 {len(documents[:top_k])} 个相似实体")

            return documents[:top_k]

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def search_by_keyword(self, keyword: str, limit: int = 50) -> List[Document]:
        """
        根据关键字搜索节点，返回Document列表

        Args:
            keyword: 搜索关键字
            limit: 返回结果数量限制

        Returns:
            Document对象列表
        """
        if not self.driver:
            logger.warning("Neo4j驱动未连接")
            return []

        try:
            with self.driver.session(database=self.database) as session:
                # 查询匹配的节点
                res = session.run(
                    """
                    MATCH (n)
                    WHERE n.name CONTAINS $kw
                    RETURN elementId(n) as node_id, n.name as name, labels(n)[0] as type, properties(n) as props
                    LIMIT $limit
                    """, kw=keyword, limit=limit
                )

                documents = []
                for rec in res:
                    # 将节点信息转换为Document
                    content = f"{rec['type']}: {rec['name']}"
                    metadata = {
                        "node_id": rec['node_id'],
                        "entity_name": rec['name'],
                        "entity_type": rec['type'],
                        "source": "knowledge_graph"
                    }
                    # 添加节点属性到metadata
                    if rec['props']:
                        metadata.update({k: v for k, v in rec['props'].items() if k != 'name'})

                    documents.append(Document(page_content=content, metadata=metadata))

                logger.info(f"图谱检索找到 {len(documents)} 个节点")
                if documents:
                    logger.debug(f"检索到的节点: {[d.metadata['entity_name'] for d in documents[:3]]}")
                return documents
        except Exception as e:
            logger.error(f"图谱检索失败: {e}")
            return []

    def search_by_relation(self, entity_name: str, limit: int = 50) -> List[Document]:
        """
        根据实体名称查询相关关系，返回Document列表

        Args:
            entity_name: 实体名称
            limit: 返回结果数量限制

        Returns:
            Document对象列表，包含关系信息
        """
        if not self.driver:
            logger.warning("Neo4j驱动未连接")
            return []

        try:
            with self.driver.session(database=self.database) as session:
                # 查询实体的所有关系
                res = session.run(
                    """
                    MATCH (a {name:$entity_name})-[r]->(b)
                    RETURN a.name as source, type(r) as relation, b.name as target,
                           labels(a)[0] as source_type, labels(b)[0] as target_type
                    LIMIT $limit
                    """, entity_name=entity_name, limit=limit
                )

                documents = []
                for rec in res:
                    # 将关系信息转换为Document
                    content = f"{rec['source_type']}: {rec['source']} -> {rec['relation']} -> {rec['target_type']}: {rec['target']}"
                    metadata = {
                        "source": rec['source'],
                        "source_type": rec['source_type'],
                        "relation": rec['relation'],
                        "target": rec['target'],
                        "target_type": rec['target_type'],
                        "data_source": "knowledge_graph"
                    }

                    documents.append(Document(page_content=content, metadata=metadata))

                logger.info(f"图谱关系检索找到 {len(documents)} 条关系")
                return documents
        except Exception as e:
            logger.error(f"图谱关系检索失败: {e}")
            return []

    def query_by_entities(self, entity_ids: List[str], depth: int = 2) -> List[Dict]:
        """
        根据实体ID查询相关子图

        Args:
            entity_ids: 实体ID列表
            depth: 查询深度

        Returns:
            关系结果列表
        """
        if not self.driver or not entity_ids:
            return []

        try:
            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH path = (start)-[rel*..{depth}]-(end)
                WHERE elementId(start) IN $entity_ids
                WITH nodes(path) AS nodes, relationships(path) AS rels
                UNWIND nodes AS node
                UNWIND rels AS rel

                WITH DISTINCT rel, startNode(rel) AS start, endNode(rel) AS end
                RETURN start.name AS source,
                       labels(start)[0] AS source_type,
                       type(rel) AS relationship,
                       end.name AS target,
                       labels(end)[0] AS target_type,
                       properties(rel) AS rel_properties
                ORDER BY source, relationship, target
                LIMIT 100
                """

                result = session.run(query, entity_ids=entity_ids)
                records = [dict(record) for record in result]
                result.consume()
                logger.info(f"子图查询找到 {len(records)} 条关系")
                return records
        except Exception as e:
            logger.error(f"子图查询失败: {e}")
            return []

    def search_graph_by_query(self,
                              query_text: str,
                              entity_types: List[str] = None,
                              relation_types: List[str] = None,
                              depth: int = 2,
                              similarity_threshold: float = 0.5,
                              top_k: int = 5) -> Dict:
        """
        根据查询文本检索知识图谱，返回格式化的文档结果

        步骤：
        1. 提取查询中的实体
        2. 向量检索相似实体
        3. 知识图谱查询
        4. 组织结果为文档格式

        Args:
            query_text: 查询文本
            entity_types: 实体类型列表（可选）
            relation_types: 关系类型列表（可选）
            depth: 查询深度
            similarity_threshold: 相似度阈值
            top_k: 返回结果数量

        Returns:
            {
                "vdb_results": [],  # 查询到的实体关系列表
                "document": "",     # 组织好的文档字符串
                "metadata": {}      # 元数据信息
            }
        """
        start_time = time.time()
        logger.info(f"处理查询: {query_text}")

        # 1. 提取查询中的实体
        entity_types = entity_types or kg_schema.get_entity_types()
        relation_types = relation_types or kg_schema.get_relationship_types()

        entities = self._extract_entities_from_query(query_text, entity_types, relation_types)
        logger.info(f"提取到 {len(entities)} 个实体")
        for e in entities:
            logger.debug(f"  - {e.get('type', '未知')}: {e.get('name', '未知')}")

        # 2. 向量检索相似实体
        entity_texts = [
            f"{e.get('type', '实体')}: {e['name']}"
            for e in entities
        ]

        if not entity_texts:
            logger.warning("没有提取到实体，使用查询文本进行检索")
            entity_texts = [query_text]

        all_similar_entity_ids = set()
        for entity_text in entity_texts:
            logger.debug(f"搜索相似实体: {entity_text}")
            similar_entities = self._search_similar_entities(
                entity_text,
                threshold=similarity_threshold,
                top_k=top_k
            )

            logger.debug(f"  找到 {len(similar_entities)} 个相似实体")
            for entity in similar_entities:
                all_similar_entity_ids.add(entity["id"])

        logger.info(f"找到 {len(all_similar_entity_ids)} 个相似实体")

        # 3. 知识图谱查询
        kg_results = []
        if all_similar_entity_ids:
            kg_results = self.query_by_entities(
                list(all_similar_entity_ids),
                depth=depth
            )

        logger.info(f"查询到 {len(kg_results)} 条关系")

        # 4. 组织结果为文档格式
        vdb_results = self._format_kg_results(kg_results)
        content = self._format_results_as_document(vdb_results, top_k=top_k)

        processing_time = time.time() - start_time

        return {
            "vdb_results": vdb_results,
            "content": content,
            "metadata": {
                "query": query_text,
                "similar_entities_count": len(all_similar_entity_ids),
                "relations_count": len(kg_results),
                "processing_time": processing_time
            }
        }

    def _extract_entities_from_query(self,
                                     query_text: str,
                                     entity_types: List[str],
                                     relation_types: List[str]) -> List[Dict]:
        """
        从查询文本中提取实体

        Args:
            query_text: 查询文本
            entity_types: 实体类型列表
            relation_types: 关系类型列表（未使用，保留用于未来扩展）

        Returns:
            实体列表
        """
        """
        从查询文本中提取实体

        Args:
            query_text: 查询文本
            entity_types: 实体类型列表
            relation_types: 关系类型列表

        Returns:
            实体列表
        """
        # 简化实现：如果没有配置LLM服务，使用简单的关键词匹配
        # 如果需要完整的实体提取，需要集成LLM服务

        # 如果向量检索不可用，尝试从查询中提取潜在的实体关键词
        entities = []

        # 简单的分词处理，提取可能的实体名
        # 移除标点符号并分割
        clean_text = query_text.replace('？', '').replace('?', '').replace('。', '').replace('.', '')
        clean_text = clean_text.replace('的', '').replace('是', '').replace('什么', '')
        words = clean_text.split()

        for word in words:
            if len(word) >= 2:  # 只保留长度>=2的词
                entities.append({
                    "name": word,
                    "type": entity_types[0] if entity_types else "实体"
                })

        return entities

    def _search_similar_entities(self,
                                 entity_text: str,
                                 threshold: float,
                                 top_k: int) -> List[Dict]:
        """
        搜索相似实体

        Args:
            entity_text: 实体文本
            threshold: 相似度阈值
            top_k: 返回数量

        Returns:
            相似实体列表
        """
        if not self.embedding_client or len(self.entity_index["embeddings"]) == 0:
            return []

        try:
            # 生成查询嵌入
            query_embedding = self.embedding_client.embed_query(entity_text)
            query_vector = np.array(query_embedding).reshape(1, -1)

            # 计算相似度
            similarities = cosine_similarity(query_vector, self.entity_index["embeddings"])[0]

            entities = []
            for idx, sim in enumerate(similarities):
                if sim >= threshold:
                    entities.append({
                        "id": self.entity_index["ids"][idx],
                        "name": self.entity_index["names"][idx],
                        "type": self.entity_index["types"][idx],
                        "similarity": float(sim)
                    })

            # 按相似度排序并返回top_k
            entities.sort(key=lambda x: x["similarity"], reverse=True)
            return entities[:top_k]

        except Exception as e:
            logger.error(f"搜索相似实体失败: {e}")
            return []

    def _format_kg_results(self, kg_results: List[Dict]) -> List[str]:
        """
        格式化知识图谱查询结果

        Args:
            kg_results: 知识图谱查询结果

        Returns:
            格式化后的结果字符串列表
        """
        vdb_results = []
        for result in kg_results[:10]:
            source_type = result.get("source_type", "")
            source = result.get("source", "")
            relationship = result.get("relationship", "")
            target_type = result.get("target_type", "")
            target = result.get("target", "")

            formatted = f"{source_type}: {source} -> {relationship} -> {target_type}: {target}"
            vdb_results.append(formatted)

        # 去重
        vdb_results = list(dict.fromkeys(vdb_results))

        return vdb_results

    def _format_results_as_document(self, vdb_results: List[str], top_k: int) -> str:
        """
        将查询结果组织成文档格式

        Args:
            vdb_results: 查询结果列表

        Returns:
            格式化的文档字符串
        """
        if not vdb_results:
            return ""

        vdb_desc = "相关文档片段：\n"
        for i, doc in enumerate(vdb_results[:top_k]):
            vdb_desc += f"片段{i + 1}: {doc[:200]}...\n"
        return vdb_desc


def get_gs(config: Dict[str, Any] = None) -> GraphSearcher:
    """
    获取图谱检索器实例（单例模式）

    Args:
        config: 配置字典，包含 neo4j 和 embedding 配置

    Returns:
        图谱检索器实例
    """
    global _gs_instance
    if not _gs_instance:
        if config is None:
            raise ValueError("首次调用必须传入config参数")

        # 提取Neo4j配置
        neo4j_config = {
            "uri": config.get("neo4j", {}).get("uri", "bolt://localhost:7687"),
            "user": config.get("neo4j", {}).get("user", "neo4j"),
            "password": config.get("neo4j", {}).get("password", ""),
            "database": config.get("neo4j", {}).get("database", "neo4j")
        }

        # 提取嵌入配置
        embedding_config = {
            "provider": config.get("embedding", {}).get("text_dense", {}).get("provider"),
            "model": config.get("embedding", {}).get("text_dense", {}).get("model"),
            "api_key": config.get("embedding", {}).get("text_dense", {}).get("api_key"),
            "base_url": config.get("embedding", {}).get("text_dense", {}).get("base_url")
        }

        _gs_instance = GraphSearcher(Neo4jConnection(neo4j_config), embedding_config=embedding_config)
    return _gs_instance
