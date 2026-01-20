"""
Neo4j数据库连接模块
负责初始化和管理Neo4j数据库连接
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from neo4j import GraphDatabase, basic_auth

logger = logging.getLogger(__name__)

# 添加当前项目目录到 Python 路径（支持直接运行）
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# 导入配置和工具函数
try:
    # 尝试相对导入（当作为包导入时）
    from ...config.models import AppConfig
except ImportError:
    # 回退到绝对导入（当直接运行文件时）
    from config.models import AppConfig


class Neo4jConnection:
    """
    Neo4j数据库连接管理类
    负责建立、维护和关闭数据库连接
    """

    def __init__(self, config: AppConfig):
        """
        初始化Neo4j连接

        Args:
            config: 应用配置
        """
        self.config = config
        self.driver: Optional[GraphDatabase.driver] = None

        # 从配置中获取Neo4j连接参数
        self.uri = getattr(config.neo4j, 'uri', 'bolt://localhost:7687') if hasattr(config, 'neo4j') else 'bolt://localhost:7687'
        self.user = getattr(config.neo4j, 'user', 'neo4j') if hasattr(config, 'neo4j') else 'neo4j'
        self.password = getattr(config.neo4j, 'password', '') if hasattr(config, 'neo4j') else ''
        self.database = getattr(config.neo4j, 'database', 'neo4j') if hasattr(config, 'neo4j') else 'neo4j'
        self.max_connection_lifetime = getattr(config.neo4j, 'max_connection_lifetime', 3600) if hasattr(config, 'neo4j') else 3600
        self.max_connection_pool_size = getattr(config.neo4j, 'max_connection_pool_size', 50) if hasattr(config, 'neo4j') else 50
        self.connection_timeout = getattr(config.neo4j, 'connection_timeout', 30.0) if hasattr(config, 'neo4j') else 30.0

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
