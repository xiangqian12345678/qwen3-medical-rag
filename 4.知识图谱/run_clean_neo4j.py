"""
Neo4j数据库清理工具
提供安全的数据库清理功能，包括删除数据、重置索引等
"""
from neo4j_connection import Neo4jConnection


class Neo4jCleaner:
    """Neo4j数据库清理类"""

    def __init__(self, connection: Neo4jConnection = None):
        """
        初始化清理工具

        【输入示例】
        cleaner = Neo4jCleaner(conn)

        【输出示例】
        None (清理工具已初始化)
        """
        self.connection = connection or Neo4jConnection()
        if not self.connection.connect():
            raise Exception("数据库连接失败")
        self.driver = self.connection.get_driver()
        self.database = self.connection.database

    def get_statistics(self) -> dict:
        """
        获取数据库统计信息

        【输出示例】
        {
            "entities": 4905,
            "relationships": 14918,
            "labels": ["药物", "症状", "疾病", "Entity"],
            "relationship_types": ["治疗", "导致", "属于"]
        }
        """
        with self.driver.session(database=self.database) as session:
            # 节点统计
            result = session.run("MATCH (n) RETURN count(n) as count")
            entity_count = result.single()["count"]

            # 关系统计
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]

            '''
            标签统计
            1. CALL db.labels() - 调用系统过程
                CALL - 调用 Neo4j 内置过程的命令
                db.labels() - 系统内置过程，返回数据库中所有存在的标签
                这是 Neo4j 的管理函数，用于元数据查询
                类似数据库的 SHOW TABLES 命令
            2. YIELD label - 提取返回值
                YIELD - 从过程返回的结果中提取特定字段
                label - db.labels() 过程返回的字段名
                该过程返回单行单列的数据，每行包含一个标签名称
            3. RETURN collect(label) as labels - 聚合返回
                collect(label) - 聚合函数，将所有标签收集到一个数组中
                as labels - 将结果数组命名为 labels
                返回格式：["Entity", "疾病", "症状", "药物"]
            '''
            result = session.run(
                """
                CALL db.labels() YIELD label
                RETURN collect(label) as labels
                """
            )
            labels = result.single()["labels"]

            # 关系类型统计
            result = session.run(
                """
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN collect(relationshipType) as relationship_types
                """
            )
            rel_types = result.single()["relationship_types"]

            '''
            Label 的结构:
            在 Neo4j 中，每个节点有两个标签：
            1. Entity - 固定标签，所有实体都有
            2. {entity_type} - 动态标签，如 药物、症状、疾病
                # neo4j_operations.py:68
                MERGE (e:Entity:{entity_type} {name: $name})
                #         ↑      ↑
                #         固定   动态
            例子：
            1. 创建药物节点
                ops.create_entity(
                    name="阿司匹林",
                    entity_type="药物",
                    properties={"成分": "乙酰水杨酸", "剂量": "100mg"}
                )
                节点表示：
                (阿司匹林:Entity:药物 {name: "阿司匹林", 成分: "乙酰水杨酸", 剂量: "100mg"})
                           ↑      ↑
                          固定   动态label
            2.2：创建症状节点
                ops.create_entity(
                    name="发热",
                    entity_type="症状",
                    properties={"描述": "体温升高", "严重程度": "轻度"}
                )
                节点表示：
                (发热:Entity:症状 {name: "发热", 描述: "体温升高", 严重程度: "轻度"})
                      ↑      ↑
                     固定   动态label
            '''
            return {
                "entities": entity_count,
                "relationships": rel_count,
                "labels": labels,
                "relationship_types": rel_types
            }

    def delete_all_data(self, confirm: bool = False) -> bool:
        """
        删除所有数据（节点和关系）

        【输入示例】
        success = cleaner.delete_all_data(confirm=True)

        【输出示例】
        ✅ 成功删除 4905 个节点和 14918 个关系
        返回: True

        ⚠️ 警告：此操作不可逆，请谨慎使用！
        """
        if not confirm:
            print("⚠️ 警告：此操作将删除所有数据，请设置 confirm=True 确认")
            return False

        stats = self.get_statistics()
        print(f"\n即将删除以下数据:")
        print(f"  节点数: {stats['entities']}")
        print(f"  关系数: {stats['relationships']}")
        print(f"  节点类型: {len(stats['labels'])}种")
        print(f"  关系类型: {len(stats['relationship_types'])}种")

        with self.driver.session(database=self.database) as session:
            # 删除所有节点和关系
            result = session.run(
                """
                MATCH (n)
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
            )
            deleted_count = result.single()["deleted_count"]

            print(f"\n✅ 成功删除 {deleted_count} 个节点")
            print(f"✅ 所有关系也已删除")

            return True

    def delete_by_label(self, label: str, confirm: bool = False) -> bool:
        """
        删除指定类型的所有节点

        【Label 说明】
        Neo4j 节点有两个 Label：
        1. "Entity" - 固定标签（所有实体都有）
        2. "{entity_type}" - 动态标签（如 "药物"、"症状"、"疾病"）

        完整节点示例：(阿司匹林:Entity:药物 {name: "阿司匹林"})
                             ↑      ↑
                            固定   动态label

        【支持删除的 Label 类型】
        - "药物" - 删除所有药物节点
        - "症状" - 删除所有症状节点
        - "疾病" - 删除所有疾病节点
        - "Entity" - 删除所有实体节点（慎用！会删除所有节点）

        【输入示例】
        success = cleaner.delete_by_label("药物", confirm=True)

        【输出示例】
        即将删除 1500 个类型为'药物'的节点
        ✅ 成功删除 1500 个类型为"药物"的节点
        返回: True

        【注意事项】
        1. DETACH DELETE 会同时删除节点及其所有关系
        2. confirm=True 必须设置为 True 才能执行删除
        3. 使用 "Entity" 标签会删除所有节点，请谨慎操作
        """
        if not confirm:
            print(f"⚠️ 警告：此操作将删除所有类型为'{label}'的节点，请设置 confirm=True 确认")
            return False

        with self.driver.session(database=self.database) as session:
            '''
            【参数说明】
                label: 要统计的节点标签（Label），如 "药物"、"症状"、"疾病"、"Entity"
            【查询详解】
                1. MATCH (n:{label})
                   - MATCH: 匹配数据库中的节点
                   - n: 节点变量名（别名），后续引用该节点时使用
                   - {label}: 节点标签，通过 f-string 动态插入标签名称
                     例如: MATCH (n:药物) 表示查找所有标签为 "药物" 的节点
    
                2. RETURN count(n) as count
                   - count(n): Neo4j 聚合函数，统计匹配到的节点数量
                   - as count: 将结果字段命名为 "count"，便于后续获取
                   - 返回格式: {count: 1500}
            '''
            result = session.run(
                f"""
                MATCH (n:{label})
                RETURN count(n) as count
                """
            )
            count = result.single()["count"]
            print(f"\n即将删除 {count} 个类型为'{label}'的节点")

            # 执行删除操作
            # DETACH DELETE: 删除节点及其所有关系
            result = session.run(
                f"""
                MATCH (n:{label})
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
            )
            deleted_count = result.single()["deleted_count"]

            print(f"✅ 成功删除 {deleted_count} 个类型为'{label}'的节点")
            return True

    def clear_embeddings(self, confirm: bool = False) -> bool:
        """
        清除所有嵌入向量（保留节点和关系）

        【输入示例】
        success = cleaner.clear_embeddings(confirm=True)

        【输出示例】
        ✅ 成功清除 4905 个实体的嵌入向量
        ✅ 成功清除 14918 个关系的嵌入向量
        返回: True
        """
        if not confirm:
            print("⚠️ 警告：此操作将清除所有嵌入向量，请设置 confirm=True 确认")
            return False

        stats = self.get_statistics()
        print(f"\n即将清除以下数据的嵌入向量:")
        print(f"  节点数: {stats['entities']}")
        print(f"  关系数: {stats['relationships']}")

        with self.driver.session(database=self.database) as session:
            # 清除实体嵌入
            result = session.run(
                """
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                REMOVE n.embedding
                RETURN count(n) as count
                """
            )
            entity_count = result.single()["count"]
            print(f"\n✅ 成功清除 {entity_count} 个实体的嵌入向量")

            # 清除关系嵌入
            result = session.run(
                """
                MATCH ()-[r]->()
                WHERE r.embedding IS NOT NULL
                REMOVE r.embedding
                RETURN count(r) as count
                """
            )
            rel_count = result.single()["count"]
            print(f"✅ 成功清除 {rel_count} 个关系的嵌入向量")

            return True

    def delete_orphan_nodes(self, confirm: bool = False) -> bool:
        """
        删除孤立节点（没有关系的节点）

        【输入示例】
        success = cleaner.delete_orphan_nodes(confirm=True)

        【输出示例】
        ✅ 成功删除 120 个孤立节点
        返回: True
        """
        if not confirm:
            print("⚠️ 警告：此操作将删除所有孤立节点，请设置 confirm=True 确认")
            return False

        with self.driver.session(database=self.database) as session:
            # 先统计
            result = session.run(
                """
                MATCH (n)
                WHERE NOT (n)-[]-()
                RETURN count(n) as count
                """
            )
            count = result.single()["count"]
            print(f"\n即将删除 {count} 个孤立节点")

            '''
            1.基础模式
                (n)              # 节点 n
                (n)-()           # n 有关系指向某个节点（匿名关系）
                (n)-[]-()        # n 通过任意类型的关系连接到某个节点【无向】
                                 # 匹配 (n)->(m) 或 (n)<-(m)
                (n)-[]->()       # n 通过任意关系指向某个节点【单向，正向】
                (n)<-[]-()       # 某节点通过任意关系指向 n【单向，反向】
            2.指令类型
                (n)-[:治疗]-(m)  # n 通过"治疗"关系连接到 m【无向】
                                # 匹配 (n)-[:治疗]->(m) 或 (n)<-[:治疗]-(m)
                (n)-[:治疗]->(m) # n 通过"治疗"关系指向 m【单向，正向】
                (n)<-[:治疗]-(m) # m 通过"治疗"关系指向 n【单向，反向】
            3.双向关系
                (n)-[]->(m) AND (n)<-[]-(m)  # n 和 m 之间有双向关系
                # 同时存在 (n)->(m) 和 (n)<-(m)
            4.多关系
                (n)-[]->(m) OR (n)<-[]-(m)  # n 和 m 之间有任意关系
                # 存在 (n)->(m) 或 (n)<-(m)    
            5.多关系类型
                (n)-[:治疗|预防|治疗并预防]->(m)  # n 和 m 之间有治疗或预防关系
                # 存在 (n)-[:治疗]->(m) 或 (n)-[:预防]->(m)
            6.NOT 否定
                WHERE NOT (n)-[]-()  # n 没有任何方向的关系（孤立节点）
                                     # 既没有 (n)->(m)，也没有 (m)->(n)
                WHERE (n)-[]-()      # n 至少有一个方向的关系（非孤立节点）
                                     # 存在 (n)->(m) 或 (m)->(n) 之一
            '''
            # 删除
            result = session.run(
                """
                MATCH (n)
                WHERE NOT (n)-[]-()
                DELETE n
                RETURN count(n) as deleted_count
                """
            )
            deleted_count = result.single()["deleted_count"]

            print(f"✅ 成功删除 {deleted_count} 个孤立节点")
            return True

    def reset_database(self, confirm: bool = False) -> bool:
        """
        重置数据库：删除所有数据并重建索引

        【输入示例】
        success = cleaner.reset_database(confirm=True)

        【输出示例】
        ✅ 已删除所有数据
        ✅ 索引已重建
        返回: True

        ⚠️ 警告：此操作不可逆，将删除所有数据和索引！
        """
        if not confirm:
            print("⚠️ 警告：此操作将重置数据库，请设置 confirm=True 确认")
            return False

        # 删除所有数据
        self.delete_all_data(confirm=True)

        # 重建索引（可选，根据需要）
        print("\n重建索引...")
        with self.driver.session(database=self.database) as session:
            # 删除旧索引
            result = session.run("SHOW INDEXES")
            for record in result:
                index_name = record["name"]
                if index_name != "LOOKUP INDEX":
                    session.run(f"DROP INDEX {index_name} IF EXISTS")

            '''
            创建基本索引
            1.CREATE INDEX - 创建索引的命令
            2.entity_name_index - 索引的名称（自定义标识符）
                entity_name_index 是一个自定义的索引名称，值就是字符串 "entity_name_index"。
            3.IF NOT EXISTS - 条件判断
                如果该索引不存在，则创建
                如果已存在，则跳过，不会报错
                这是一个幂等操作，可以安全地重复执行
            4.FOR (n:Entity) - 指定索引作用的节点类型
                n 是节点变量的别名
                :Entity 是节点标签（label）
                表示这个索引只应用于带有 Entity 标签的节点
            5.ON (n.name) - 指定索引作用的属性
                n.name 表示节点 n 的 name 属性
                索引会针对 name 属性的值建立
                
            索引类型：
                | 类型      | 语法关键字             | 场景          |
                | -------- | --------------------- | ------------ |
                | RANGE    | CREATE INDEX          | 等值/范围查询  |
                | TEXT     | CREATE TEXT INDEX     | 字符串模糊     |
                | POINT    | CREATE POINT INDEX    | 地理位置       |
                | FULLTEXT | CREATE FULLTEXT INDEX | 搜索引擎场景    |
                | LOOKUP   | 系统自带                | id/label 查询 |
            '''
            session.run(
                """
                CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)
                """
            )
            print("✅ 索引已重建")

        return True

    def close(self):
        """关闭连接"""
        self.connection.close()


def main():
    """主函数：交互式清理工具"""
    print("=" * 60)
    print("Neo4j 数据库清理工具")
    print("=" * 60)

    try:
        # 初始化清理工具
        print("\n🔌 连接数据库...")
        cleaner = Neo4jCleaner()
        print("✅ 连接成功")

        # 显示当前统计信息
        print("\n" + "=" * 60)
        print("当前数据库状态")
        print("=" * 60)
        stats = cleaner.get_statistics()
        print(f"\n📊 统计信息:")
        print(f"  节点数: {stats['entities']}")
        print(f"  关系数: {stats['relationships']}")
        print(f"\n  节点类型 ({len(stats['labels'])}种):")
        for label in sorted(stats['labels']):
            print(f"    - {label}")
        print(f"\n  关系类型 ({len(stats['relationship_types'])}种):")
        for rel_type in sorted(stats['relationship_types']):
            print(f"    - {rel_type}")

        # 显示操作菜单
        print("\n" + "=" * 60)
        print("可用操作")
        print("=" * 60)
        print("1. 删除所有数据（节点和关系）")
        print("2. 删除指定类型的节点")
        print("3. 清除所有嵌入向量")
        print("4. 删除孤立节点")
        print("5. 重置数据库（删除所有数据并重建索引）")
        print("6. 仅显示统计信息")
        print("0. 退出")
        print("=" * 60)

        # 获取用户输入
        choice = input("\n请选择操作 (0-6): ").strip()

        if choice == "1":
            print("\n⚠️ 严重警告：此操作将删除所有数据！")
            confirm = input("确认删除？(输入 'yes' 确认): ").strip().lower()
            if confirm == "yes":
                cleaner.delete_all_data(confirm=True)
            else:
                print("❌ 操作已取消")

        elif choice == "2":
            label = input("请输入要删除的节点类型: ").strip()
            print(f"\n⚠️ 警告：此操作将删除所有类型为'{label}'的节点！")
            confirm = input("确认删除？(输入 'yes' 确认): ").strip().lower()
            if confirm == "yes":
                cleaner.delete_by_label(label, confirm=True)
            else:
                print("❌ 操作已取消")

        elif choice == "3":
            print("\n⚠️ 警告：此操作将清除所有嵌入向量！")
            print("提示：修改嵌入模型配置后需要执行此操作")
            confirm = input("确认清除？(输入 'yes' 确认): ").strip().lower()
            if confirm == "yes":
                cleaner.clear_embeddings(confirm=True)
            else:
                print("❌ 操作已取消")

        elif choice == "4":
            print("\n⚠️ 警告：此操作将删除所有孤立节点！")
            confirm = input("确认删除？(输入 'yes' 确认): ").strip().lower()
            if confirm == "yes":
                cleaner.delete_orphan_nodes(confirm=True)
            else:
                print("❌ 操作已取消")

        elif choice == "5":
            print("\n⚠️ 严重警告：此操作将重置数据库！")
            print("将删除所有数据和索引，此操作不可逆！")
            confirm = input("确认重置？(输入 'yes' 确认): ").strip().lower()
            if confirm == "yes":
                cleaner.reset_database(confirm=True)
            else:
                print("❌ 操作已取消")

        elif choice == "6":
            print("\n✅ 仅显示统计信息，不执行任何操作")

        elif choice == "0":
            print("\n👋 退出")

        else:
            print("\n❌ 无效的选择")

        # 关闭连接
        cleaner.close()
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
