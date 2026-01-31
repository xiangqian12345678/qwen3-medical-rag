"""
网络搜索模块测试文件
基于 search/web_search.py 的 main 函数逻辑
"""
import json
import logging
import time

from langchain_core.documents import Document

from rag.rag_loader import RAGConfigLoader
from recall.search import create_web_search_tool
from recall.search.web_search import get_ws, WebSearchState
from recall.search.web_searcher import reset_kb
from utils import create_llm_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebSearchTester:
    """网络搜索测试类"""

    def __init__(self):
        """初始化测试环境"""
        self.llm = None
        self.search_tool = None
        self.search_llm = None
        self.search_cnt = 10

    def setup(self):
        """设置测试环境"""
        logger.info("=" * 60)
        logger.info("初始化网络搜索测试环境")
        logger.info("=" * 60)

        try:
            # 初始化 WebSearcher
            reset_kb()
            get_ws({})
            logger.info("✓ WebSearcher 初始化成功")

            # 初始化LLM
            rag_config = RAGConfigLoader().config
            power_model = create_llm_client(rag_config.llm)
            self.llm = power_model
            logger.info("✓ LLM初始化成功")

            # 创建网络搜索工具
            self.search_tool, self.search_llm = create_web_search_tool(
                search_cnt=self.search_cnt,
                power_model=self.llm
            )

            if self.search_tool is None:
                logger.warning("⚠ 网络搜索工具创建失败")
                return False

            logger.info("✓ 网络搜索工具创建成功")
            return True

        except Exception as e:
            logger.error(f"✗ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_direct_web_search(self, query: str = "人工智能在医疗领域的应用"):
        """测试1: 直接调用网络搜索工具"""
        logger.info("\n" + "=" * 60)
        logger.info("测试1: 直接网络搜索")
        logger.info("=" * 60)

        try:
            logger.info(f"搜索查询: {query}")

            # 直接调用搜索工具
            result = self.search_tool.invoke(query)
            logger.info(f"✓ 网络搜索执行成功")
            logger.info(f"  结果长度: {len(result)} 字符")

            # 解析并显示结果
            results_data = json.loads(result)
            logger.info(f"  检索到 {len(results_data)} 条结果")

            if results_data:
                logger.info(f"\n前3条结果预览:")
                for i, item in enumerate(results_data[:3], 1):
                    logger.info(f"\n--- 结果 {i} ---")
                    logger.info(f"  内容: {item['page_content'][:500]}...")
                    logger.info(f"  来源: {item['metadata'].get('source', 'N/A')}")
            else:
                logger.warning("⚠ 未检索到任何结果")

            return True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_multi_web_search(self, queries: list = None):
        """测试4: 批量网络搜索"""
        if queries is None:
            queries = [
                "新冠疫苗的最新进展",
                "人工智能在医疗领域的应用",
                "心脏病的早期症状"
            ]

        logger.info("\n" + "=" * 60)
        logger.info(f"测试4: 批量网络搜索（{len(queries)}个查询）")
        logger.info("=" * 60)

        try:
            results = []

            for i, query in enumerate(queries, 1):
                logger.info(f"\n查询 {i}/{len(queries)}: {query}")

                try:
                    result = self.search_tool.invoke(query)
                    result_data = json.loads(result)
                    results.append((query, len(result_data)))
                    logger.info(f"  ✓ 检索到 {len(result_data)} 条结果")
                except Exception as e:
                    logger.error(f"  ✗ 检索失败: {e}")
                    results.append((query, 0))

            logger.info(f"\n批量搜索完成")
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
        logger.info("开始运行网络搜索测试套件")
        logger.info("=" * 60)

        # 设置测试环境
        if not self.setup():
            logger.error("测试环境设置失败，退出测试")
            return

        results = {
            "测试1-直接网络搜索": self.test_direct_web_search(),
            "测试2-批量网络搜索": self.test_multi_web_search(),
        }

        # 每次测试之间暂停5秒
        test_names = list(results.keys())
        for i in range(len(test_names) - 1):
            logger.info(f"\n等待5秒后继续...")
            time.sleep(5)

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
    tester = WebSearchTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
