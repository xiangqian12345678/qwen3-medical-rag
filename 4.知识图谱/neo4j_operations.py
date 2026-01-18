"""
Neo4j数据操作模块
提供基础的数据库操作功能
"""
from typing import Dict, List, Optional
from neo4j_connection import Neo4jConnection


class Neo4jOperations:
    """
    Neo4j数据操作类
    提供创建、查询、更新等基础操作
    """

    def __init__(self, connection: Neo4jConnection = None, database: str = None):
        """
        初始化数据操作对象

        【输入示例】
        conn = Neo4jConnection()
        conn.connect()
        ops = Neo4jOperations(conn)

        【输出示例】
        None (操作对象已初始化)
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

    def create_entity(self, name: str, entity_type: str,
                      properties: Dict = None) -> Optional[int]:
        """
        创建实体节点

        【输入示例】
        entity_id = ops.create_entity(
            name="阿司匹林",
            entity_type="药物",
            properties={"成分": "乙酰水杨酸", "剂量": "100mg"}
        )

        【输出示例】
        123  # 返回实体内部ID
        """
        if not self.driver:
            print("❌ 数据库未连接")
            return None

        properties = properties or {}

        try:
            with self.driver.session(database=self.database) as session:
                # 构建 Cypher 查询语句
                # MERGE: 如果节点存在则匹配，不存在则创建（相当于"创建或更新"操作）
                # (e:Entity:{entity_type}): e是节点别名，Entity是固定标签，{entity_type}是动态标签（如"药物"、"症状"）
                # {name: $name}: 根据name属性匹配或创建节点，$name是参数化变量（防止注入）
                # SET e += $props: 合并属性，不会覆盖已有属性，只添加/更新传入的属性
                # RETURN elementId(e) as id: 返回节点的唯一ID（Neo4j 5.x+ 语法）
                query = f"""
                MERGE (e:Entity:{entity_type} {{name: $name}})
                SET e += $props
                RETURN elementId(e) as id
                """

                # 执行查询，传入参数：name是节点名称，props是额外属性字典
                result = session.run(query, name=name, props=properties)

                # 获取单条结果记录
                record = result.single()

                # 消耗结果集（释放资源）
                result.consume()

                # 如果查询成功，返回节点ID
                if record:
                    return record["id"]
                return None
        except Exception as e:
            print(f"❌ 创建实体失败: {e}")
            return None

    def create_relationship(self, source_id: int, target_id: int,
                            rel_type: str, properties: Dict = None) -> Optional[int]:
        """
        创建关系

        【输入示例】
        rel_id = ops.create_relationship(
            source_id=123,
            target_id=456,
            rel_type="治疗",
            properties={"强度": "强"}
        )

        【输出示例】
        789  # 返回关系内部ID
        """
        if not self.driver:
            print("❌ 数据库未连接")
            return None

        properties = properties or {}

        try:
            with self.driver.session(database=self.database) as session:
                # 构建 Cypher 查询语句
                # MATCH (source), (target): 匹配两个独立的节点，分别命名为 source 和 target
                # WHERE: 过滤条件，根据节点ID精确定位两个节点
                # elementId(source)/elementId(target): 获取节点的唯一ID（Neo4j 5.x+ 语法）
                # MERGE (source)-[r:{rel_type}]->(target): 创建或更新关系
                #   - [r:{rel_type}]: r是关系别名，{rel_type}是动态关系类型（如"治疗"、"导致"）
                #   - ->: 表示有向关系，从source指向target
                #   - 如果该类型的关系已存在则匹配，不存在则创建
                # SET r += $props: 合并关系属性，不覆盖已有属性
                # RETURN elementId(r) as id: 返回关系的唯一ID
                query = f"""
                MATCH (source), (target)
                WHERE elementId(source) = $source_id AND elementId(target) = $target_id
                MERGE (source)-[r:{rel_type}]->(target)
                SET r += $props
                RETURN elementId(r) as id
                """

                # 执行查询，传入参数：源节点ID、目标节点ID、关系属性字典
                result = session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    props=properties
                )

                # 获取单条结果记录
                record = result.single()

                # 消耗结果集（释放资源）
                result.consume()

                # 如果查询成功，返回关系ID
                if record:
                    return record["id"]
                return None
        except Exception as e:
            print(f"❌ 创建关系失败: {e}")
            return None

    def get_entity_by_id(self, entity_id: int) -> Optional[Dict]:
        """
        根据ID获取实体

        【输入示例】
        entity = ops.get_entity_by_id(123)

        【输出示例】
        {
            "id": 123,
            "name": "阿司匹林",
            "type": "药物",
            "properties": {"成分": "乙酰水杨酸"}
        }
        """
        if not self.driver:
            return None

        try:
            with self.driver.session() as session:
                query = """
                MATCH (n)
                WHERE elementId(n) = $id
                RETURN elementId(n) as id, n.name as name,
                       labels(n)[0] as type, properties(n) as properties
                """

                result = session.run(query, id=entity_id)
                record = result.single()
                result.consume()

                if record:
                    return {
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "properties": record["properties"] or {}
                    }
                return None
        except Exception as e:
            print(f"❌ 获取实体失败: {e}")
            return None

    def get_entity_by_name(self, name: str) -> Optional[Dict]:
        """
        根据名称获取实体

        【输入示例】
        entity = ops.get_entity_by_name("阿司匹林")

        【输出示例】
        {
            "id": 123,
            "name": "阿司匹林",
            "type": "药物",
            "properties": {"成分": "乙酰水杨酸"}
        }
        """
        if not self.driver:
            return None

        try:
            with self.driver.session() as session:
                query = """
                MATCH (n)
                WHERE n.name = $name
                RETURN id(n) as id, n.name as name,
                       labels(n)[0] as type, properties(n) as properties
                LIMIT 1
                """

                result = session.run(query, name=name)
                record = result.single()
                result.consume()

                if record:
                    return {
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "properties": record["properties"] or {}
                    }
                return None
        except Exception as e:
            print(f"❌ 获取实体失败: {e}")
            return None

    def update_entity(self, entity_id: int, properties: Dict) -> bool:
        """
        更新实体属性

        【输入示例】
        success = ops.update_entity(
            entity_id=123,
            properties={"剂量": "200mg", "注意": "饭后服用"}
        )

        【输出示例】
        True  # 更新成功
        """
        if not self.driver:
            return False

        try:
            with self.driver.session() as session:
                query = """
                MATCH (n) WHERE id(n) = $id
                SET n += $props
                """

                result = session.run(query, id=entity_id, props=properties)
                result.consume()
                return True
        except Exception as e:
            print(f"❌ 更新实体失败: {e}")
            return False

    def delete_entity(self, entity_id: int) -> bool:
        """
        删除实体及其所有关系

        【输入示例】
        success = ops.delete_entity(123)

        【输出示例】
        True  # 删除成功
        """
        if not self.driver:
            return False

        try:
            with self.driver.session() as session:
                query = """
                MATCH (n) WHERE id(n) = $id
                DETACH DELETE n
                """

                result = session.run(query, id=entity_id)
                result.consume()
                return True
        except Exception as e:
            print(f"❌ 删除实体失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, int]:
        """
        获取数据库统计信息

        【输入示例】
        stats = ops.get_statistics()

        【输出示例】
        {
            "entities": 150,
            "relationships": 320
        }
        """
        stats = {"entities": 0, "relationships": 0}

        if not self.driver:
            return stats

        try:
            with self.driver.session() as session:
                # 查询实体数量
                result = session.run("MATCH (e) RETURN count(e) as count")
                record = result.single()
                result.consume()
                stats["entities"] = record["count"] if record else 0

                # 查询关系数量
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = result.single()
                result.consume()
                stats["relationships"] = record["count"] if record else 0

        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")

        return stats


# 使用示例
if __name__ == "__main__":
    # 示例1: 创建实体
    print("示例1: 创建实体")
    conn = Neo4jConnection()
    conn.connect()
    ops = Neo4jOperations(conn)

    entity_id = ops.create_entity(
        name="阿司匹林",
        entity_type="药物",
        properties={"成分": "乙酰水杨酸", "剂量": "100mg"}
    )
    print(f"创建实体，ID: {entity_id}")

    # 示例2: 查询实体
    print("\n示例2: 查询实体")
    entity = ops.get_entity_by_id(entity_id)
    print(f"实体: {entity}")

    # 示例3: 更新实体
    print("\n示例3: 更新实体")
    ops.update_entity(entity_id, {"注意": "饭后服用"})
    entity = ops.get_entity_by_id(entity_id)
    print(f"更新后: {entity}")

    # 示例4: 创建关系
    print("\n示例4: 创建关系")
    symptom_id = ops.create_entity("头痛", "症状", {})
    if symptom_id:
        rel_id = ops.create_relationship(
            source_id=entity_id,
            target_id=symptom_id,
            rel_type="治疗",
            properties={}
        )
        print(f"创建关系，ID: {rel_id}")

    # 示例5: 获取统计信息
    print("\n示例5: 获取统计信息")
    stats = ops.get_statistics()
    print(f"实体数: {stats['entities']}")
    print(f"关系数: {stats['relationships']}")

    conn.close()
