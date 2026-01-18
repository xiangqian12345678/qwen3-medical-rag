"""
Neo4j数据保存模块
负责将提取的实体和关系保存到数据库
"""
from typing import Dict, List
from neo4j_connection import Neo4jConnection
from embedding_service import EmbeddingService
from config import kg_schema


class Neo4jSave:
    """
    Neo4j数据保存类
    负责批量保存实体和关系，并生成嵌入向量
    """

    def __init__(self, connection: Neo4jConnection = None,
                 embedding_service: EmbeddingService = None,
                 database: str = None):
        """
        初始化保存对象

        【输入示例】
        conn = Neo4jConnection()
        conn.connect()
        embed_service = EmbeddingService()
        saver = Neo4jSave(conn, embed_service)

        【输出示例】
        None (保存对象已初始化)
        """
        self.connection = connection or Neo4jConnection()
        if database:
            self.connection.database = database
        if self.connection.check_connection():
            self.driver = self.connection.get_driver()
        else:
            self.driver = None

        self.embedding_service = embedding_service or EmbeddingService()
        self.entity_cache = {}

        # 获取数据库名称
        self.database = self.connection.database

    def save_entities_and_relationships(self,
                                        entities: List[Dict],
                                        relationships: List[Dict]) -> bool:
        """
        批量保存实体和关系

        【输入示例】
        entities = [
            {"id": "e1", "name": "阿司匹林", "type": "药物", "properties": {}},
            {"id": "e2", "name": "头痛", "type": "症状", "properties": {}}
        ]
        relationships = [
            {"source": "e1", "target": "e2", "type": "治疗", "properties": {}}
        ]
        success = saver.save_entities_and_relationships(entities, relationships)

        【输出示例】
        ✅ 成功保存 2 个实体和 1 个关系
        返回: True
        """
        if not self.driver:
            print("❌ 数据库未连接")
            return False

        try:
            with self.driver.session(database=self.database) as session:
                # 保存实体
                entity_id_map = {}
                for entity in entities:
                    entity_id = self._save_entity(session, entity)
                    if entity_id:
                        entity_id_map[entity["id"]] = entity_id
                        self.entity_cache[entity_id] = {
                            "name": entity["name"],
                            "type": entity["type"]
                        }

                # 保存关系
                for rel in relationships:
                    source_id = entity_id_map.get(rel["source"])
                    target_id = entity_id_map.get(rel["target"])

                    if source_id and target_id:
                        self._save_relationship(session, rel, source_id, target_id)

                print(f"✅ 成功保存 {len(entities)} 个实体和 {len(relationships)} 个关系")
                return True
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False

    def _save_entity(self, session, entity: Dict) -> int:
        """
        保存单个实体

        【输入示例】
        entity_id = saver._save_entity(session, {
            "id": "e1",
            "name": "阿司匹林",
            "type": "药物",
            "properties": {}
        })

        【输出示例】
        123  # 返回实体内部ID
        """
        entity_name = entity["name"]
        entity_type = entity["type"]
        properties = entity.get("properties", {})

        '''
        构建查询
        1. MERGE - 合并操作
            MERGE 是 Neo4j 的特殊操作，等同于"创建或匹配"
            如果节点已存在（基于匹配条件），则返回该节点
            如果节点不存在，则创建新节点
            这是"Upsert"（更新或插入）操作，确保数据不重复
        2. (e:Entity:{entity_type} {name: $name}) - 节点模式
            e - 节点变量名（别名），后续引用该节点时使用
            :Entity - 节点的标签，所有实体都有 Entity 标签
            :{entity_type} - 动态标签，例如 :疾病、:症状、:药物 等
            {name: $name} - 匹配/创建条件，基于节点的 name 属性
            $name 是参数化查询，防止注入攻击
        3. SET e += $props - 设置属性
            e += $props 将参数中的所有属性添加到节点 e
            使用 += 运算符表示"添加或更新"属性
            不会删除已有属性，只会添加新属性或更新已有属性
        4. RETURN elementId(e) as id - 返回结果
            elementId(e) 获取节点的唯一标识符
            as id 将返回值命名为 id
            用于确认操作是否成功并获取节点 ID
        '''
        query = f"""
        MERGE (e:Entity:{entity_type} {{name: $name}})
        SET e += $props
        RETURN elementId(e) as id
        """

        result = session.run(query, name=entity_name, props=properties)
        record = result.single()
        result.consume()

        if record:
            entity_id = record["id"]

            # 生成并保存嵌入向量
            entity_text = f"{entity_type}: {entity_name}"
            embedding = self.embedding_service.generate_embedding(entity_text)

            if embedding:
                session.run(
                    """
                    MATCH (e) WHERE elementId(e) = $id
                    SET e.embedding = $embedding
                    """, id=entity_id, embedding=embedding
                )

            return entity_id
        return None

    def _save_relationship(self, session, rel: Dict, source_id: int, target_id: int) -> int:
        """
        保存单个关系

        【输入示例】
        rel_id = saver._save_relationship(session, rel, 123, 456)

        【输出示例】
        789  # 返回关系内部ID
        """
        rel_type = rel["type"]
        properties = rel.get("properties", {})

        # 从缓存获取实体名称
        source_name = self.entity_cache.get(source_id, {}).get("name", "Unknown")
        target_name = self.entity_cache.get(target_id, {}).get("name", "Unknown")

        '''
        1. MATCH (source), (target) - 匹配节点
            查找两个节点，分别命名为 source（源节点）和 target（目标节点）
            这里没有指定匹配条件，条件在 WHERE 子句中
            逗号表示同时查找多个节点（等同于独立查找）
        2. WHERE elementId(source) = $source_id AND elementId(target) = $target_id - 过滤条件
            elementId(source) - 获取源节点的唯一标识符
            elementId(target) - 获取目标节点的唯一标识符
            $source_id 和 $target_id 是参数，传入节点的 ID
            AND 表示两个条件必须同时满足
            确保准确定位到特定的两个节点
        3. MERGE (source)-[r:{rel_type}]->(target) - 创建/合并关系
            MERGE - 创建或匹配关系（类似节点的 MERGE）
            (source) - 源节点（已通过 MATCH 找到）
            [r:{rel_type}] - 关系变量和类型
            r - 关系变量名（别名）
            :{rel_type} - 关系类型（动态），如 :导致、:治疗、:症状
            -> - 有向关系，从 source 指向 target
            (target) - 目标节点（已通过 MATCH 找到）
            如果关系已存在，则返回；否则创建新关系
        4. SET r += $props - 设置关系属性
            r += $props 将参数中的属性添加到关系
            += 运算符：添加新属性或更新已有属性
            不会删除现有属性
        5. RETURN - 返回结果
            返回空（当前代码不完整，通常会返回 r 或关系信息）
        '''
        query = f"""
        MATCH (source), (target)
        WHERE elementId(source) = $source_id AND elementId(target) = $target_id
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $props
        RETURN elementId(r) as id
        """

        result = session.run(
            query,
            source_id=source_id,
            target_id=target_id,
            props=properties
        )
        record = result.single()
        result.consume()

        if record:
            rel_id = record["id"]

            # 生成并保存嵌入向量
            rel_text = f"{rel_type}: {source_name} -> {target_name}"
            embedding = self.embedding_service.generate_embedding(rel_text)

            if embedding:
                session.run(
                    """
                    MATCH ()-[r]->() WHERE elementId(r) = $id
                    SET r.embedding = $embedding
                    """, id=rel_id, embedding=embedding
                )

            return rel_id
        return None

    def save_text_knowledge(self, text: str,
                            llm_service,
                            entity_types: List[str] = None,
                            relation_types: List[str] = None) -> bool:
        """
        从文本提取并保存知识

        【输入示例】
        text = "阿司匹林可以治疗头痛和发热"
        success = saver.save_text_knowledge(text, llm_service)
        或
        success = saver.save_text_knowledge(
            text, llm_service,
            entity_types=["药物", "症状", "疾病"],
            relation_types=["治疗", "导致"]
        )

        【输出示例】
        ✅ 提取到 3 个实体和 2 个关系
        ✅ 成功保存 3 个实体和 2 个关系
        返回: True
        """
        # 如果没有指定实体类型和关系类型，从kg_schema.json读取
        if entity_types is None:
            entity_types = kg_schema.get_entity_types()
        if relation_types is None:
            relation_types = kg_schema.get_relationship_types()

        # 提取实体和关系
        result = llm_service.extract_entities_relations(
            text, entity_types, relation_types
        )

        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        if not entities and not relationships:
            print("⚠️ 未提取到任何实体或关系")
            return False

        print(f"✅ 提取到 {len(entities)} 个实体和 {len(relationships)} 个关系")

        # 保存到数据库
        return self.save_entities_and_relationships(entities, relationships)


# 使用示例
if __name__ == "__main__":
    from llm_service import LLMService

    # 示例1: 保存实体和关系
    print("示例1: 保存实体和关系")
    conn = Neo4jConnection()
    conn.connect()

    embed_service = EmbeddingService()
    saver = Neo4jSave(conn, embed_service)

    entities = [
        {"id": "e1", "name": "阿司匹林", "type": "药物", "properties": {"成分": "乙酰水杨酸"}},
        {"id": "e2", "name": "头痛", "type": "症状", "properties": {}}
    ]
    relationships = [
        {"source": "e1", "target": "e2", "type": "治疗", "properties": {}}
    ]

    saver.save_entities_and_relationships(entities, relationships)

    # 示例2: 从文本提取并保存
    print("\n示例2: 从文本提取并保存")
    llm_service = LLMService()
    text = "阿司匹林是一种常用药物，主要用于治疗头痛和发热。"

    saver.save_text_knowledge(
        text,
        llm_service,
        entity_types=["药物", "症状", "疾病"],
        relation_types=["治疗", "导致"]
    )

    llm_service.close()
    embed_service.close()
    conn.close()
