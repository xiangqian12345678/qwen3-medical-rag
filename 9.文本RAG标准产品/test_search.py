"""
网络搜索模块测试文件
基于 search/web_search.py 的 main 函数逻辑
"""
import json
import logging
import sys
from pathlib import Path

from rag_loader import RAGConfigLoader
from utils import create_llm_client

# 添加项目路径
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from langchain_core.documents import Document

from recall.search import create_web_search_tool, llm_network_search
from recall.search.web_search import get_ws, reset_kb

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
        self.tool_node = None
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
            self.search_tool, self.search_llm, self.tool_node = create_web_search_tool(
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

    def test_direct_web_search(self, query: str = "阿司匹林副作用"):
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

    def test_llm_network_search_node(self):
        """测试2: 测试llm_network_search节点"""
        logger.info("\n" + "=" * 60)
        logger.info("测试2: LLM网络搜索节点")
        logger.info("=" * 60)

        try:
            # 模拟输入状态
            test_state = {
                "query": "阿司匹林有哪些副作用？",
                "docs": [
                    Document(
                        page_content="阿司匹林是一种非甾体抗炎药，主要用于解热镇痛。",
                        metadata={"source": "medical_db_1"}
                    )
                ],
                "main_messages": [],
                "other_messages": [],
                "answer": "",
                "retry": 0,
                "final": "",
                "judge_result": ""
            }

            logger.info(f"查询问题: {test_state['query']}")
            logger.info(f"输入文档数: {len(test_state['docs'])}")
            logger.info(f"输入文档内容: {test_state['docs'][0].page_content}")

            # 执行网络搜索节点
            result_state = llm_network_search(
                state=test_state,
                judge_llm=self.llm,
                network_search_llm=self.search_llm,
                network_tool_node=self.tool_node,
                show_debug=True
            )

            logger.info(f"✓ llm_network_search节点执行成功")
            logger.info(f"  输出文档数: {len(result_state.get('docs', []))}")
            logger.info(f"  其他消息数: {len(result_state.get('other_messages', []))}")

            if len(result_state.get('docs', [])) > 0:
                logger.info(f"\n第一个文档内容: {result_state.get('docs', [])[0].page_content[:200]}...")

            return True

        except Exception as e:
            logger.error(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_llm_judgment_no_search_needed(self):
        """测试3: 测试LLM判断不需要网络搜索的情况"""
        logger.info("\n" + "=" * 60)
        logger.info("测试3: LLM判断无需网络搜索")
        logger.info("=" * 60)

        try:
            from recall.search import get_prompt_template
            from langchain_core.messages import HumanMessage, SystemMessage

            # 模拟已有充足文档的情况
            test_query = "普通感冒的治疗方法"
            existing_docs = """
            文档1: 普通感冒是一种常见的上呼吸道感染，主要由病毒引起。症状包括鼻塞、流涕、咳嗽、咽痛等。
            文档2: 感冒的治疗主要是对症治疗，包括多休息、多饮水、使用解热镇痛药缓解发热和头痛。
            文档3: 感冒一般一周左右可自愈，无需特殊治疗。如症状加重或持续不退，应及时就医。
            """

            test_state = {
                "query": test_query,
                "docs": [
                    Document(page_content=doc, metadata={"source": f"doc_{i + 1}"})
                    for i, doc in enumerate(existing_docs.split("\n文档") if existing_docs else [])
                ],
                "main_messages": [],
                "other_messages": [],
                "answer": "",
                "retry": 0,
                "final": "",
                "judge_result": ""
            }

            logger.info(f"查询问题: {test_query}")
            logger.info(f"已有文档数: {len(test_state['docs'])}")

            # 执行网络搜索节点
            result_state = llm_network_search(
                state=test_state,
                judge_llm=self.llm,
                network_search_llm=self.search_llm,
                network_tool_node=self.tool_node,
                show_debug=True
            )

            logger.info(f"✓ 判断执行完成")
            logger.info(f"  最终文档数: {len(result_state.get('docs', []))}")
            logger.info(f"  其他消息数: {len(result_state.get('other_messages', []))}")

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

    def test_complete_workflow(self):
        """测试5: 完整工作流测试"""
        logger.info("\n" + "=" * 60)
        logger.info("测试5: 完整工作流测试")
        logger.info("=" * 60)

        try:
            # 模拟一个完整的RAG流程中的网络搜索环节
            test_state = {
                "query": "最新的癌症治疗药物有哪些？",
                "docs": [
                    Document(
                        page_content="癌症治疗包括手术、放疗、化疗等多种方式。",
                        metadata={"source": "medical_db_1"}
                    ),
                    Document(
                        page_content="传统的化疗药物有很多副作用。",
                        metadata={"source": "medical_db_2"}
                    )
                ],
                "main_messages": [],
                "other_messages": [],
                "answer": "",
                "retry": 0,
                "final": "",
                "judge_result": ""
            }

            logger.info("场景: 用户询问最新的癌症治疗药物，但本地数据库只有基础信息")
            logger.info(f"查询问题: {test_state['query']}")
            logger.info(f"已有文档数: {len(test_state['docs'])}")

            # 执行网络搜索节点
            result_state = llm_network_search(
                state=test_state,
                judge_llm=self.llm,
                network_search_llm=self.search_llm,
                network_tool_node=self.tool_node,
                show_debug=True
            )

            logger.info(f"\n工作流执行结果:")
            logger.info(f"  输出文档数: {len(result_state.get('docs', []))}")
            logger.info(f"  其他消息数: {len(result_state.get('other_messages', []))}")

            # 分析文档变化
            doc_count_before = len(test_state['docs'])
            doc_count_after = len(result_state.get('docs', []))

            if doc_count_after > doc_count_before:
                logger.info(f"  文档增加: {doc_count_after - doc_count_before} 条")
            elif doc_count_after < doc_count_before:
                logger.info(f"  文档减少: {doc_count_before - doc_count_after} 条")
            else:
                logger.info(f"  文档数量不变")

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
            "测试2-LLM网络搜索节点": self.test_llm_network_search_node(),
            "测试3-LLM判断无需搜索": self.test_llm_judgment_no_search_needed(),
            "测试4-批量网络搜索": self.test_multi_web_search(),
            "测试5-完整工作流": self.test_complete_workflow()
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
    tester = WebSearchTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
