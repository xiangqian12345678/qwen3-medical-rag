"""
Milvus向量检索模块测试文件
基于 milvus/embed_search.py 的 main 函数逻辑
"""
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from rag.rag_loader import RAGConfigLoader
from recall.milvus.embed_loader import EmbedConfigLoader
from recall.milvus.embed_search import create_db_search_tool
from utils import create_llm_client, create_embedding_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MilvusSearchTester:
    """向量数据库检索测试类"""

    def __init__(self):
        """初始化测试环境"""
        self.embedConfigLoader = None
        self.llm = None
        self.db_search_tool = None
        self.db_search_llm = None
        self.db_tool_node = None

    def setup(self):
        """设置测试环境"""
        logger.info("=" * 60)
        logger.info("初始化向量数据库测试环境")
        logger.info("=" * 60)

        try:
            # 加载配置
            self.embedConfigLoader = EmbedConfigLoader()
            logger.info(f"✓ 配置加载成功")
            logger.info(f"  Milvus URI: {self.embedConfigLoader.milvus.uri}")
            logger.info(f"  集合名称: {self.embedConfigLoader.milvus.collection_name}")

            # 初始化LLM
            rag_config = RAGConfigLoader().config
            power_model = create_llm_client(rag_config.llm)
            self.llm = power_model
            logger.info("✓ LLM初始化成功")

            # 初始化向量模型
            embed_model = create_embedding_client(rag_config.embedding)

            # 创建数据库检索工具
            self.db_search_tool, self.db_search_llm = create_db_search_tool(
                embed_config_loader=self.embedConfigLoader,
                power_model=self.llm,
                embed_model=embed_model
            )
            logger.info("✓ 数据库检索工具创建成功")

            return True

        except Exception as e:
            logger.error(f"✗ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_direct_tool_invocation(self, query: str = "高血压的治疗方法"):
        """测试1: 直接调用数据库检索工具"""
        logger.info("\n" + "=" * 60)
        logger.info("测试1: 直接调用检索工具")
        logger.info("=" * 60)

        try:
            logger.info(f"检索查询: {query}")

            # 直接调用工具
            tool_result = self.db_search_tool.invoke({"query": query})
            result_data = json.loads(tool_result)

            logger.info(f"✓ 工具调用成功")
            logger.info(f"  返回结果数量: {len(result_data)}")

            if result_data:
                logger.info(f"\n前3条结果预览:")
                for i, item in enumerate(result_data[:3], 1):
                    logger.info(f"\n--- 结果 {i} ---")
                    logger.info(f"  内容: {item['page_content'][:500]}...")
                    logger.info(f"  元数据: {item['metadata']}")
            else:
                logger.warning("⚠ 未检索到任何结果")

            return True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False



    def test_multi_query(self, queries: list = None):
        """测试4: 批量查询测试"""
        if queries is None:
            queries = [
                "糖尿病的病因是什么？",
                "高血压的治疗药物有哪些？",
                "冠心病的早期症状"
            ]

        logger.info("\n" + "=" * 60)
        logger.info(f"测试4: 批量查询测试（{len(queries)}个查询）")
        logger.info("=" * 60)

        try:
            results = []

            for i, query in enumerate(queries, 1):
                logger.info(f"\n查询 {i}/{len(queries)}: {query}")

                try:
                    tool_result = self.db_search_tool.invoke({"query": query})
                    result_data = json.loads(tool_result)
                    results.append((query, len(result_data)))
                    logger.info(f"  ✓ 检索到 {len(result_data)} 条结果")
                except Exception as e:
                    logger.error(f"  ✗ 检索失败: {e}")
                    results.append((query, 0))

            logger.info(f"\n批量查询完成")
            for query, count in results:
                logger.info(f"  {query}: {count} 条结果")

            return True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


    def run_all_tests(self):
        """运行所有测试"""
        logger.info("\n" + "=" * 60)
        logger.info("开始运行向量数据库检索测试套件")
        logger.info("=" * 60)

        # 设置测试环境
        if not self.setup():
            logger.error("测试环境设置失败，退出测试")
            return

        results = {
            "测试1-直接工具调用": self.test_direct_tool_invocation(),
            "测试2-批量查询": self.test_multi_query(),
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


def main():
    """主函数"""
    tester = MilvusSearchTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
