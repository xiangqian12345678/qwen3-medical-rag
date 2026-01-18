"""
Neo4j数据库连接模块
负责初始化和管理Neo4j数据库连接
"""
from typing import Optional
from neo4j import GraphDatabase, basic_auth
from config import neo4j_config


class Neo4jConnection:
    """
    Neo4j数据库连接管理类
    负责建立、维护和关闭数据库连接
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        """
        初始化Neo4j连接

        【输入示例】
        conn = Neo4jConnection(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="12345678",
            database="neo4j"
        )

        【输出示例】
        ✅ Neo4j连接成功
        """
        self.uri = uri or neo4j_config.uri
        self.user = user or neo4j_config.user
        self.password = password or neo4j_config.password
        # 默认使用 neo4j 数据库,避免创建数据库的问题
        self.database = database or neo4j_config.database
        self.driver: Optional[GraphDatabase.driver] = None

    def connect(self) -> bool:
        """
        连接到Neo4j数据库
        如果指定的数据库不存在,会尝试创建(如果支持)

        【输入示例】
        conn.connect()

        【输出示例】
        True  # 连接成功
        """
        try:
            # 创建数据库驱动
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.user, self.password),
                max_connection_lifetime=neo4j_config.max_connection_lifetime,
                max_connection_pool_size=neo4j_config.max_connection_pool_size,
                connection_timeout=neo4j_config.connection_timeout
            )

            # 直接连接到目标数据库进行测试
            # 不再尝试创建数据库,因为 Neo4j 社区版不支持
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run("RETURN 'connection_test' AS test")
                    record = result.single()
                    result.consume()

                    if record and record["test"] == "connection_test":
                        print(f"✅ Neo4j连接成功 (数据库: {self.database})")
                        return True
            except Exception as test_error:
                # 如果连接失败,检查错误信息
                error_msg = str(test_error)
                if "database" in error_msg.lower() and "not found" in error_msg.lower():
                    print(f"❌ 数据库 '{self.database}' 不存在")
                    print(f"💡 提示: 请先在 Neo4j 中创建该数据库,或者使用默认的 'neo4j' 数据库")
                else:
                    print(f"❌ 连接测试失败: {test_error}")
                return False

            print("❌ Neo4j连接测试失败")
            return False
        except Exception as e:
            print(f"❌ Neo4j连接失败: {e}")
            print(f"💡 请检查:")
            print(f"  1. Neo4j 服务是否启动 (bolt://localhost:7687)")
            print(f"  2. 用户名和密码是否正确 (用户: {self.user})")
            print(f"  3. 数据库 '{self.database}' 是否存在")
            return False

    def get_driver(self):
        """
        获取数据库驱动对象

        【输入示例】
        driver = conn.get_driver()

        【输出示例】
        <neo4j.GraphDatabase.driver object>
        """
        return self.driver

    def close(self):
        """
        关闭数据库连接

        【输入示例】
        conn.close()

        【输出示例】
        None (连接已关闭)
        """
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j连接已关闭")

    def check_connection(self) -> bool:
        """
        检查连接是否正常

        【输入示例】
        is_connected = conn.check_connection()

        【输出示例】
        True  # 连接正常
        """
        if not self.driver:
            return False

        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 as ping").single()
            return True
        except Exception:
            return False

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 使用示例
if __name__ == "__main__":
    # 示例1: 基本连接
    print("示例1: 基本连接")
    conn = Neo4jConnection()
    if conn.connect():
        print("连接成功!")
        conn.close()

    # 示例2: 使用上下文管理器
    print("\n示例2: 使用上下文管理器")
    with Neo4jConnection() as conn:
        if conn.check_connection():
            print("连接状态: 正常")

    # 示例3: 检查连接
    print("\n示例3: 检查连接")
    conn = Neo4jConnection()
    conn.connect()
    print(f"连接状态: {'正常' if conn.check_connection() else '异常'}")
    conn.close()
