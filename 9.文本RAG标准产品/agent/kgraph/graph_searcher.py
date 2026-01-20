"""
知识图谱检索模块
提供基于Neo4j的图谱查询功能
"""
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

# 添加当前模块目录到 Python 路径（支持直接运行）
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    # 尝试相对导入（当作为包导入时）
    from .neo4j_connection import Neo4jConnection
except ImportError:
    # 回退到直接导入（当直接运行文件时）
    from neo4j_connection import Neo4jConnection

logger = logging.getLogger(__name__)

_gs_instance: Optional["GraphSearcher"] = None


class GraphSearcher:
    """
    图谱检索类
    提供基于关键词的图谱检索功能，将查询结果转换为Document对象
    """

    def __init__(self, connection: Neo4jConnection, database: str = None):
        """
        初始化图谱检索器

        Args:
            connection: Neo4j连接对象
            database: 数据库名称（可选）
        """
        connected = connection.connect()
        if not connected:
            logger.warning(f"Neo4j连接失败: {neo4j_conn.uri}")

        self.connection = connection
        if database:
            self.connection.database = database
        if self.connection.check_connection():
            self.driver = self.connection.get_driver()
        else:
            self.driver = None

        # 获取数据库名称
        self.database = self.connection.database

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
                        "source": "knowledge_graph"
                    }

                    documents.append(Document(page_content=content, metadata=metadata))

                logger.info(f"图谱关系检索找到 {len(documents)} 条关系")
                return documents
        except Exception as e:
            logger.error(f"图谱关系检索失败: {e}")
            return []

    def search_graph(self, keyword: str, limit: int = 50) -> List[Document]:
        """
        综合图谱检索：同时检索实体和关系

        Args:
            keyword: 搜索关键字
            limit: 返回结果数量限制

        Returns:
            Document对象列表
        """
        documents = []

        # 检索匹配的实体
        entity_docs = self.search_by_keyword(keyword, limit)
        documents.extend(entity_docs)

        # 检索相关关系
        relation_docs = self.search_by_relation(keyword, limit)
        documents.extend(relation_docs)

        # 去重（基于page_content）
        seen_contents = set()
        unique_documents = []
        for doc in documents:
            if doc.page_content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(doc.page_content)

        return unique_documents[:limit]


def get_gs(config: Dict[str, Any] = None) -> GraphSearcher:
    """
    获取图谱检索器实例（单例模式）

    Args:
        config: 配置字典

    Returns:
        图谱检索器实例
    """
    global _gs_instance
    if not _gs_instance:
        _gs_instance = GraphSearcher(Neo4jConnection(**config))
    return _gs_instance
