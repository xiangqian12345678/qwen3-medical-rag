"""
Neo4j数据库连接模块
负责初始化和管理Neo4j数据库连接
"""
import sys
from pathlib import Path
from typing import Optional

from neo4j import GraphDatabase, basic_auth



class Neo4jConnection:
    """
    Neo4j数据库连接管理类
    负责建立、维护和关闭数据库连接
    """

    def __init__(self, neo4jConfig):
        """
        初始化Neo4j连接

        Args:
            neo4jConfig: 应用配置（可以是字典、对象或配置模块）
        """
        self.neo4jConfig = neo4jConfig
        self.driver: Optional[GraphDatabase.driver] = None

        # 支持字典和对象两种配置方式
        if isinstance(neo4jConfig, dict):
            self.uri = neo4jConfig.get('uri', 'bolt://localhost:7687')
            self.user = neo4jConfig.get('user', 'neo4j')
            self.password = neo4jConfig.get('password', '')
            self.database = neo4jConfig.get('database', 'neo4j')
            self.max_connection_lifetime = neo4jConfig.get('max_connection_lifetime', 3600)
            self.max_connection_pool_size = neo4jConfig.get('max_connection_pool_size', 50)
            self.connection_timeout = neo4jConfig.get('connection_timeout', 30.0)
        elif hasattr(neo4jConfig, 'neo4j'):
            # 从配置对象中获取Neo4j连接参数
            self.uri = getattr(neo4jConfig.neo4j, 'uri', 'bolt://localhost:7687')
            self.user = getattr(neo4jConfig.neo4j, 'user', 'neo4j')
            self.password = getattr(neo4jConfig.neo4j, 'password', '')
            self.database = getattr(neo4jConfig.neo4j, 'database', 'neo4j')
            self.max_connection_lifetime = getattr(neo4jConfig.neo4j, 'max_connection_lifetime', 3600)
            self.max_connection_pool_size = getattr(neo4jConfig.neo4j, 'max_connection_pool_size', 50)
            self.connection_timeout = getattr(neo4jConfig.neo4j, 'connection_timeout', 30.0)
        else:
            # 默认配置
            self.uri = 'bolt://localhost:7687'
            self.user = 'neo4j'
            self.password = ''
            self.database = 'neo4j'
            self.max_connection_lifetime = 3600
            self.max_connection_pool_size = 50
            self.connection_timeout = 30.0

    def connect(self) -> bool:
        """
        连接到Neo4j数据库
        如果指定的数据库不存在,会尝试创建(如果支持)

        Returns:
            bool: 连接是否成功
        """
        try:
            # 创建数据库驱动
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.user, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size,
                connection_timeout=self.connection_timeout
            )

            # 连接到目标数据库进行测试
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run("RETURN 'connection_test' AS test")
                    record = result.single()
                    result.consume()

                    if record and record["test"] == "connection_test":
                        return True
            except Exception as test_error:
                # 如果连接失败,检查错误信息
                error_msg = str(test_error)
                if "database" in error_msg.lower() and "not found" in error_msg.lower():
                    logger.error(f"数据库 '{self.database}' 不存在")
                else:
                    logger.error(f"连接测试失败: {test_error}")
                return False

            logger.error("Neo4j连接测试失败")
            return False
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return False

    def get_driver(self):
        """
        获取数据库驱动对象

        Returns:
            Neo4j driver对象
        """
        return self.driver

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()

    def check_connection(self) -> bool:
        """
        检查连接是否正常

        Returns:
            bool: 连接是否正常
        """
        if not self.driver:
            return False

        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 as ping").single()
            return True
        except Exception:
            return False
