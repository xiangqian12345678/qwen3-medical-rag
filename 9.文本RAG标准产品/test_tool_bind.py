"""简单测试工具绑定和调用"""
import logging
import sys
from pathlib import Path

# 添加项目路径
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from rag_loader import RAGConfigLoader
from enhance.utils import create_llm_client
from recall.search import create_web_search_tool

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("工具绑定测试")
    logger.info("=" * 60)

    try:
        # 初始化 WebSearcher
        from recall.search.web_search import reset_kb, get_ws
        reset_kb()
        get_ws({})
        logger.info("✓ WebSearcher 初始化成功")

        # 初始化LLM
        rag_config = RAGConfigLoader().config
        power_model = create_llm_client(rag_config.llm)
        logger.info(f"✓ LLM初始化成功: {rag_config.llm.model}")

        # 创建网络搜索工具
        search_tool, search_llm = create_web_search_tool(
            search_cnt=10,
            power_model=power_model
        )

        if search_tool is None:
            logger.error("✗ 网络搜索工具创建失败")
            return

        logger.info("✓ 网络搜索工具创建成功")
        logger.info(f"  工具名称: {search_tool.name}")
        logger.info(f"  工具类型: {type(search_tool)}")

        # 测试工具调用
        logger.info("\n" + "=" * 60)
        logger.info("测试工具调用")
        logger.info("=" * 60)

        query = "人工智能在医疗领域的应用"
        logger.info(f"查询: {query}")

        # 直接调用工具
        result = search_tool.invoke(query)
        logger.info(f"✓ 直接调用成功，返回长度: {len(result)}")

        # 测试 LLM 工具调用
        logger.info("\n" + "=" * 60)
        logger.info("测试LLM工具调用")
        logger.info("=" * 60)

        from langchain_core.messages import SystemMessage, HumanMessage
        from recall.search.search_templates import get_prompt_template

        web_ai = search_llm.invoke([
            SystemMessage(content=get_prompt_template("call_web")["system"]),
            HumanMessage(content=get_prompt_template("call_web")["user"].format(search_query=query))
        ])

        logger.info(f"LLM响应: {web_ai.content[:200]}")
        logger.info(f"是否有tool_calls: {hasattr(web_ai, 'tool_calls')}")

        if hasattr(web_ai, 'tool_calls') and web_ai.tool_calls:
            logger.info(f"Tool calls: {web_ai.tool_calls}")
            for tc in web_ai.tool_calls:
                # tool_calls 可能是字典或对象
                if isinstance(tc, dict):
                    logger.info(f"  name: {tc.get('name', 'N/A')}, args: {tc.get('args', {})}, id: {tc.get('id', 'N/A')}")
                else:
                    logger.info(f"  name: {getattr(tc, 'name', 'N/A')}, args: {getattr(tc, 'args', {})}, id: {getattr(tc, 'id', 'N/A')}")

            # 直接调用工具（不使用 ToolNode）
            logger.info("\n直接调用工具...")
            try:
                if web_ai.tool_calls and len(web_ai.tool_calls) > 0:
                    tc = web_ai.tool_calls[0]
                    # tool_calls 可能是字典或对象
                    if isinstance(tc, dict):
                        query_arg = tc.get('args', {}).get('query', '')
                        tool_name = tc.get('name', 'N/A')
                    else:
                        query_arg = tc.args.get('query', '') if hasattr(tc, 'args') else ''
                        tool_name = getattr(tc, 'name', 'N/A')

                    logger.info(f"  工具名称: {tool_name}")
                    logger.info(f"  提取的 query: {query_arg}")

                    if query_arg:
                        # 使用原始的 search_tool 直接调用
                        tool_result = search_tool.invoke(query_arg)
                        logger.info(f"✓ 工具调用成功，返回长度: {len(tool_result)}")

        logger.info("\n" + "=" * 60)
        logger.info("测试完成")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
