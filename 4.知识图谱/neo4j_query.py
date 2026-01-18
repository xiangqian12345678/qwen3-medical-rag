"""
Neo4j查询模块
提供高级查询功能
"""
from typing import Dict, List
from neo4j_connection import Neo4jConnection


class Neo4jQuery:
    """
    Neo4j查询类
    提供图查询、路径查询等高级功能
    """

    def __init__(self, connection: Neo4jConnection = None, database: str = None):
        """
        初始化查询对象

        【输入示例】
        conn = Neo4jConnection()
        conn.connect()
        query = Neo4jQuery(conn)

        【输出示例】
        None (查询对象已初始化)
        """
        self.connection = connection or Neo4jConnection()
        if database:
            self.connection.database = database
        if self.connection.check_connection():
            self.driver = self.connection.get_driver()
        else:
            self.driver = None

        # 获取数据库名称
        self.database = self.connection.database

    def search_by_keyword(self, keyword: str, limit: int = 50) -> Dict:
        """
        根据关键字搜索节点

        【输入示例】
        result = query.search_by_keyword("阿司匹林", limit=20)

        【输出示例】
        {
            "nodes": [
                {"id": 123, "name": "阿司匹林", "type": "药物"},
                {"id": 456, "name": "头痛", "type": "症状"}
            ],
            "links": [
                {"source": 123, "target": 456, "type": "治疗"}
            ]
        }
        """
        if not self.driver:
            return {"nodes": [], "links": []}

        try:
            with self.driver.session(database=self.database) as session:
                # 查询匹配的节点
                res = session.run("""
                    MATCH (n)
                    WHERE n.name CONTAINS $kw
                    RETURN elementId(n) as node_id, n.name as name, labels(n)[0] as type
                    LIMIT $limit
                """, kw=keyword, limit=limit)

                nodes = []
                node_set = set()
                for rec in res:
                    node_id = rec["node_id"]
                    if node_id not in node_set:
                        nodes.append({
                            "id": node_id,
                            "name": rec["name"],
                            "type": rec["type"] or "Entity"
                        })
                        node_set.add(node_id)

                # 查询节点之间的关系
                links = []
                if node_set:
                    node_id_list = list(node_set)
                    res_links = session.run("""
                        MATCH (n)-[r]->(m)
                        WHERE elementId(n) IN $node_ids OR elementId(m) IN $node_ids
                        RETURN elementId(n) as source_id, elementId(m) as target_id, type(r) as type
                        LIMIT 100
                    """, node_ids=node_id_list)

                    for rec in res_links:
                        links.append({
                            "source": rec["source_id"],
                            "target": rec["target_id"],
                            "type": rec["type"]
                        })

                return {"nodes": nodes, "links": links}
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return {"nodes": [], "links": []}

    def shortest_path(self, source: str, target: str) -> List[Dict]:
        """
        查找两个实体之间的最短路径

        【输入示例】
        paths = query.shortest_path("阿司匹林", "感冒")

        【输出示例】
        [
            {
                "nodes": [
                    {"id": 123, "name": "阿司匹林", "type": "药物"},
                    {"id": 456, "name": "头痛", "type": "症状"},
                    {"id": 789, "name": "感冒", "type": "疾病"}
                ],
                "rels": [
                    {"source": "阿司匹林", "target": "头痛", "type": "治疗"},
                    {"source": "感冒", "target": "头痛", "type": "导致"}
                ]
            }
        ]
        """
        if not self.driver:
            return []

        try:
            with self.driver.session(database=self.database) as session:
                res = session.run("""
                    MATCH path = shortestPath(
                        (a {name:$src})-[*]-(b {name:$tgt})
                    )
                    RETURN [
                        n in nodes(path) |
                        {id:elementId(n), name:n.name, type:labels(n)[0]}
                    ] as nodes,
                    [
                        r in relationships(path) |
                        {source:startNode(r).name, target:endNode(r).name, type:type(r)}
                    ] as rels
                """, src=source, tgt=target)

                return [dict(r) for r in res]
        except Exception as e:
            print(f"❌ 最短路径查询失败: {e}")
            return []

    def query_by_entities(self, entity_ids: List[int], depth: int = 2) -> List[Dict]:
        """
        根据实体ID查询相关子图

        【输入示例】
        results = query.query_by_entities([123, 456], depth=2)

        【输出示例】
            {
                "source": "阿司匹林",
                "source_type": "药物",
                "relationship": "治疗",
                "target": "头痛",
                "target_type": "症状",
                "rel_properties": {
                    'embedding': [-0.02545723206106859, ......, -0.03518568902721352, -0.020791543516080715]
                }
            }        ]
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
                return records
        except Exception as e:
            print(f"❌ 子图查询失败: {e}")
            return []

    def query_triples(self, head: str = None, relation: str = None,
                      tail: str = None, limit: int = 50) -> Dict:
        """
        查询三元组（头-关系-尾）

        【输入示例】
        results = query.query_triples(head="阿司匹林", relation="治疗", limit=20)

        【输出示例】
        {
            "nodes": [
                {"id": 123, "name": "阿司匹林", "type": "药物"},
                {"id": 456, "name": "头痛", "type": "症状"}
            ],
            "links": [
                {"source": 123, "target": 456, "type": "治疗"}
            ]
        }
        """
        if not self.driver:
            return {"nodes": [], "links": []}

        try:
            with self.driver.session(database=self.database) as session:
                # 构建查询条件
                conditions = []
                params = {}

                if head:
                    conditions.append("h.name CONTAINS $head")
                    params["head"] = head
                if relation:
                    conditions.append("type(r) CONTAINS $relation")
                    params["relation"] = relation
                if tail:
                    conditions.append("t.name CONTAINS $tail")
                    params["tail"] = tail

                where_clause = ""
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)

                query = f"""
                MATCH (h)-[r]->(t)
                {where_clause}
                RETURN elementId(h) as source_id, h.name as source, labels(h)[0] as source_type,
                       elementId(t) as target_id, t.name as target, labels(t)[0] as target_type,
                       type(r) as type
                LIMIT $limit
                """

                params["limit"] = limit
                result = session.run(query, params)

                # 转换结果
                nodes = []
                links = []
                node_set = set()

                for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]

                    if source_id not in node_set:
                        nodes.append({
                            "id": source_id,
                            "name": record["source"],
                            "type": record["source_type"]
                        })
                        node_set.add(source_id)

                    if target_id not in node_set:
                        nodes.append({
                            "id": target_id,
                            "name": record["target"],
                            "type": record["target_type"]
                        })
                        node_set.add(target_id)

                    links.append({
                        "source": source_id,
                        "target": target_id,
                        "type": record["type"]
                    })

                return {"nodes": nodes, "links": links}
        except Exception as e:
            print(f"❌ 三元组查询失败: {e}")
            return {"nodes": [], "links": []}

    def get_all_graph(self, limit: int = 500) -> Dict:
        """
        获取整个知识图谱（限制数量）

        【输入示例】
        graph = query.get_all_graph(limit=100)

        【输出示例】
        {
            "nodes": [...],
            "links": [...]
        }
        """
        if not self.driver:
            return {"nodes": [], "links": []}

        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (start)-[r]->(end)
                RETURN elementId(start) as source_id, start.name as source, labels(start)[0] as source_type,
                       elementId(end) as target_id, end.name as target, labels(end)[0] as target_type,
                       type(r) as type
                LIMIT $limit
                """

                result = session.run(query, limit=limit)

                nodes = []
                links = []
                node_set = set()

                for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]

                    if source_id not in node_set:
                        nodes.append({
                            "id": source_id,
                            "name": record["source"],
                            "type": record["source_type"]
                        })
                        node_set.add(source_id)

                    if target_id not in node_set:
                        nodes.append({
                            "id": target_id,
                            "name": record["target"],
                            "type": record["target_type"]
                        })
                        node_set.add(target_id)

                    links.append({
                        "source": source_id,
                        "target": target_id,
                        "type": record["type"]
                    })

                return {"nodes": nodes, "links": links}
        except Exception as e:
            print(f"❌ 获取图谱失败: {e}")
            return {"nodes": [], "links": []}


# 使用示例
if __name__ == "__main__":
    # 示例1: 关键字搜索
    print("示例1: 关键字搜索")
    conn = Neo4jConnection()
    conn.connect()
    query = Neo4jQuery(conn)

    result = query.search_by_keyword("阿司匹林")
    print(f"找到 {len(result['nodes'])} 个节点")
    print(f"找到 {len(result['links'])} 条关系")

    # 示例2: 三元组查询
    print("\n示例2: 三元组查询")
    result = query.query_triples(head="阿司匹林", relation="治疗")
    print(f"找到 {len(result['links'])} 条匹配关系")

    # 示例3: 获取整个图谱
    print("\n示例3: 获取整个图谱")
    graph = query.get_all_graph(limit=10)
    print(f"图谱包含 {len(graph['nodes'])} 个节点")
    print(f"图谱包含 {len(graph['links'])} 条关系")

    conn.close()
