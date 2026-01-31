"""
知识图谱检索模块测试文件
基于 kgraph/kgraph_search.py 的 main 函数逻辑
"""
import logging
import sys
from pathlib import Path

from rag.rag_loader import RAGConfigLoader
from utils import create_llm_client
from utils import create_embedding_client

# 添加项目路径
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from langchain_core.messages import HumanMessage, SystemMessage

from recall.kgraph.kg_loader import KGraphConfigLoader
from recall.kgraph.neo4j_connection import Neo4jConnection
from recall.kgraph.kgraph_searcher import GraphSearcher
from recall.kgraph import create_kgraph_search_tool

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KGraphSearchTester:
    """知识图谱检索测试类"""

    def __init__(self):
        """初始化测试环境"""
        self.embed_model = None
        self.kgraphConfigLoader = None
        self.neo4j_conn = None
        self.graph_searcher = None
        self.llm = None
        self.kgraph_tool = None
        self.kgraph_llm = None
        self.kgraph_tool_node = None

    def setup(self):
        """设置测试环境"""
        logger.info("=" * 60)
        logger.info("初始化知识图谱测试环境")
        logger.info("=" * 60)

        try:
            # 加载配置
            self.kgraphConfigLoader = KGraphConfigLoader()
            logger.info(f"✓ 配置加载成功")
            logger.info(f"  Neo4j URI: {self.kgraphConfigLoader.neo4j_config.uri}")
            logger.info(f"  数据库: {self.kgraphConfigLoader.neo4j_config.database}")

            # 创建Neo4j连接
            self.neo4j_conn = Neo4jConnection(self.kgraphConfigLoader)
            connected = self.neo4j_conn.connect()

            if not connected:
                logger.error(f"✗ Neo4j连接失败: {self.neo4j_conn.uri}")
                return False

            logger.info("✓ Neo4j连接成功")

            # 创建向量模型
            rag_config = RAGConfigLoader().config
            self.embed_model = create_embedding_client(rag_config.embedding)

            # # 创建图谱检索器
            # embedding_config = {
            #     "provider": self.kgraphConfigLoader.get("embedding.provider", "ollama"),
            #     "model": self.kgraphConfigLoader.get("embedding.model", "nomic-embed-text"),
            #     "api_key": self.kgraphConfigLoader.get("embedding.api_key", None),
            #     "base_url": self.kgraphConfigLoader.get("embedding.base_url", "http://localhost:11434/v1")
            # }
            self.graph_searcher = GraphSearcher(self.neo4j_conn, database=self.kgraphConfigLoader.neo4j_config.database, embed_model=self.embed_model)
            logger.info("✓ 图谱检索器创建成功")

            rag_config = RAGConfigLoader().config
            power_model = create_llm_client(rag_config.llm)
            self.llm = power_model

            logger.info("✓ LLM初始化成功")

            # 创建检索工具
            self.kgraph_tool, self.kgraph_llm, self.kgraph_tool_node = create_kgraph_search_tool(
                self.kgraphConfigLoader, power_model,self.embed_model
            )

            if self.kgraph_tool is None:
                logger.warning("⚠ 图谱检索工具未启用")
            else:
                logger.info("✓ 图谱检索工具创建成功")

            return True

        except Exception as e:
            logger.error(f"✗ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_query_search(self, query: str = "房颤的治疗目的是什么？", top_k: int = 5):
        """测试1: query检索（关键词检索）"""
        logger.info("\n" + "=" * 60)
        logger.info("测试1: 关键词检索")
        logger.info("=" * 60)

        try:
            logger.info(f"搜索关键词: '{query}'")
            result = self.graph_searcher.search_graph_by_query(query, top_k=top_k)

            content = result.get("content", "")
            vdb_results = result.get("vdb_results", [])

            logger.info(f"✓ 检索成功")
            logger.info(f"  Content: {content}")
            logger.info(f"  VDB结果数: {len(vdb_results)}")

            if vdb_results:
                logger.info(f"  前3条结果预览:")
                for i, doc in enumerate(vdb_results[:3], 1):
                    logger.info(f"    {i}. {doc[:200]}...")

            return True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_relation_search(self, entity_name: str = "阿司匹林", limit: int = 5):
        """测试2: 关系检索"""
        logger.info("\n" + "=" * 60)
        logger.info("测试2: 关系检索")
        logger.info("=" * 60)

        try:
            logger.info(f"查询实体: '{entity_name}' 的关系")
            docs = self.graph_searcher.search_by_relation(entity_name, limit=limit)

            logger.info(f"✓ 找到 {len(docs)} 条关系")

            if docs:
                for i, doc in enumerate(docs[:5], 1):
                    logger.info(f"  {i}. {doc.page_content[:200]}...")

            return True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_keyword_search(self, keyword: str = "糖尿病", limit: int = 10):
        """测试3: 关键词检索（综合图谱检索）"""
        logger.info("\n" + "=" * 60)
        logger.info("测试3: 综合图谱检索")
        logger.info("=" * 60)

        try:
            logger.info(f"综合检索关键词: '{keyword}'")
            result = self.graph_searcher.search_by_keyword(keyword, limit=limit)

            logger.info(f"✓ 找到 {len(result)} 条结果（实体）")

            if result:
                for i, doc in enumerate(result[:5], 1):
                    logger.info(f"  {i}. {doc.page_content[:200]}...")

            return True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_tool_invocation(self, query: str = "高血压"):
        """测试4: 创建检索工具并调用"""
        logger.info("\n" + "=" * 60)
        logger.info("测试4: 检索工具调用")
        logger.info("=" * 60)

        try:
            if self.kgraph_tool is None:
                logger.warning("⚠ 图谱检索工具未启用，跳过此测试")
                return False

            logger.info(f"工具名称: {self.kgraph_tool.name}")
            logger.info(f"工具描述: {self.kgraph_tool.description}")

            # 执行工具调用
            logger.info(f"使用工具搜索: '{query}'")
            result = self.kgraph_tool.invoke({"query": query})

            logger.info(f"✓ 工具调用成功")
            logger.info(f"  检索结果（前500字符）:")
            logger.info(f"  {str(result)[:500]}...")

            return True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_llm_judgment(self, query: str = "心肌梗死的治疗方法"):
        """测试5: LLM判断是否需要图谱检索"""
        logger.info("\n" + "=" * 60)
        logger.info("测试5: LLM判断图谱检索")
        logger.info("=" * 60)

        try:
            from recall.kgraph.kg_templates import get_prompt_template

            # 调用LLM判断是否需要调用图谱检索工具
            kg_ai = self.llm.invoke([
                SystemMessage(content=get_prompt_template("call_kgraph")["system"]),
                HumanMessage(content=get_prompt_template("call_kgraph")["user"].format(query=query))
            ])

            logger.info(f"查询问题: {query}")
            logger.info(f"LLM响应: {kg_ai.content[:200]}...")

            # 检查是否决定调用工具
            tool_calls = getattr(kg_ai, 'tool_calls', None)
            if tool_calls and len(tool_calls) > 0:
                logger.info(f"✓ LLM决定调用图谱检索工具")
                logger.info(f"  工具调用数: {len(tool_calls)}")
            else:
                logger.info(f"✓ LLM决定不调用图谱检索工具")

            return True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self):
        """运行所有测试"""
        logger.info("\n" + "=" * 60)
        logger.info("开始运行知识图谱检索测试套件")
        logger.info("=" * 60)

        # 设置测试环境
        if not self.setup():
            logger.error("测试环境设置失败，退出测试")
            return

        results = {
            "测试1-关键词检索": self.test_query_search(),
            "测试2-关系检索": self.test_relation_search(),
            # "测试3-综合图谱检索": self.test_keyword_search(),
            "测试4-检索工具调用": self.test_tool_invocation(),
            "测试5-LLM判断检索": self.test_llm_judgment()
        }

        # 输出测试结果汇总
        logger.info("\n" + "=" * 60)
        logger.info("测试结果汇总")
        logger.info("=" * 60)

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for test_name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"  {test_name}: {status}")

        logger.info(f"\n总计: {passed}/{total} 通过")

        # 清理资源
        self.teardown()

    def teardown(self):
        """清理测试环境"""
        logger.info("\n清理测试环境...")

        if self.neo4j_conn:
            self.neo4j_conn.close()
            logger.info("✓ Neo4j连接已关闭")

        logger.info("测试环境清理完成")


def main():
    """主函数"""
    tester = KGraphSearchTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
