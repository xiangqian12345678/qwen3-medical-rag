"""网络搜索模块"""

"""网络搜索模块"""
import logging
from typing import List

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .search_templates import get_prompt_template
from .search_utils import json_to_list_document, _should_call_tool
from .web_searcher import get_ws, reset_kb


class WebSearchState(TypedDict, total=False):
    query: str
    other_messages: List
    docs: List[Document]


logger = logging.getLogger(__name__)


# =============================================================================
# 网络搜索结果模型
# =============================================================================
class NetworkSearchResult(BaseModel):
    """网络搜索结果类"""
    need_search: bool = Field(description="是否需要进行网络搜索")
    search_query: str = Field(description="网络搜索查询词", default="")
    remain_doc_index: List[int] = Field(description="保留的文档索引列表", default=[])


def llm_network_search(
        state: "WebSearchState",
        llm: BaseChatModel,
        search_tool=None,
        show_debug: bool = False
) -> "WebSearchState":
    """
    网络检索节点

    ========== 功能说明 ==========
    该节点负责：
    1. 接收用户查询，让LLM判断是否需要调用网络搜索工具
    2. 如果需要，执行网络检索并获取相关文档
    3. 将检索到的文档添加到状态中供后续RAG使用

    Args:
        state: WebSearchState状态
        llm: LLM实例
        search_tool: 网络搜索工具实例
        show_debug: 是否显示调试信息
    """
    print('-' * 60)
    print("开始网络检索")
    print('-' * 60)

    query = state["query"]

    if show_debug:
        logger.info(f"开始网络检索节点，查询: {query}")

    # 调用LLM，让其判断是否需要调用网络搜索工具
    web_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("call_web")["system"]),
        HumanMessage(content=get_prompt_template("call_web")["user"].format(search_query=query))
    ])
    state["other_messages"].append(web_ai)

    if show_debug:
        logger.info(f"LLM响应内容: {web_ai.content[:200]}")
        logger.info(f"LLM响应是否有tool_calls: {hasattr(web_ai, 'tool_calls')}")
        if hasattr(web_ai, 'tool_calls') and web_ai.tool_calls:
            logger.info(f"Tool calls数量: {len(web_ai.tool_calls)}")
            for i, tc in enumerate(web_ai.tool_calls):
                logger.info(
                    f"Tool call {i}: name={tc.get('name', 'N/A') if isinstance(tc, dict) else tc.name}, args={tc.get('args', {}) if isinstance(tc, dict) else tc.args}")
                if hasattr(tc, 'id'):
                    logger.info(f"  Tool call id: {tc.id}")

    # 检查LLM是否决定调用工具
    if _should_call_tool(web_ai):
        if show_debug:
            tool_calls = getattr(web_ai, 'tool_calls', None)
            if tool_calls and len(tool_calls) > 0:
                try:
                    if hasattr(tool_calls[0], 'args'):
                        args = tool_calls[0].args
                    elif isinstance(tool_calls[0], dict):
                        args = tool_calls[0].get('args', {})
                    else:
                        args = {}
                    logger.info(f"开始网络检索，检索参数：{args}")
                except Exception as e:
                    logger.error(f"获取工具参数失败: {e}")

        try:
            # 执行工具调用
            if show_debug:
                logger.info(f"准备调用工具，输入消息数: {len([web_ai])}")

            # 直接调用工具（不使用 ToolNode，避免配置问题）
            try:
                tool_calls = getattr(web_ai, 'tool_calls', [])
                if tool_calls and len(tool_calls) > 0:
                    tc = tool_calls[0]
                    query = tc.get('args', {}).get('query') if isinstance(tc, dict) else (
                        tc.args.get('query') if hasattr(tc, 'args') else None)
                    if query:
                        # 直接调用原始工具（使用传递的 search_tool 参数）
                        if search_tool:
                            tool_result = search_tool.invoke(query)
                            tool_message = ToolMessage(
                                content=tool_result,
                                tool_call_id=tc.get('id') if isinstance(tc, dict) else (
                                    tc.id if hasattr(tc, 'id') else 'unknown'),
                                name=tc.get('name') if isinstance(tc, dict) else (
                                    tc.name if hasattr(tc, 'name') else 'web_search')
                            )
                            tool_msgs = [tool_message]
                            state["other_messages"].append(tool_message)
                        else:
                            logger.error("search_tool 未提供，无法执行工具调用")
                            tool_msgs = []
                    else:
                        logger.error(f"无法从 tool_calls 中提取 query 参数")
                        tool_msgs = []
                else:
                    tool_msgs = []
            except Exception as e:
                logger.error(f"工具调用失败: {e}")
                tool_msgs = []

            # 检查工具返回内容
            if tool_msgs and len(tool_msgs) > 0:
                tool_content = tool_msgs[0].content
            else:
                tool_content = ""

            if show_debug:
                logger.info(f"工具返回内容长度: {len(tool_content)}")
                if tool_content:
                    logger.info(f"工具返回内容前200字符: {tool_content[:200]}")

            # 如果返回空内容,跳过JSON解析
            if not tool_content or not tool_content.strip():
                logger.warning("工具返回空内容,跳过JSON解析")
                return state

            # 将ToolMessage中的JSON字符串转换为Document对象列表
            new_docs = json_to_list_document(tool_content)
            state["docs"].extend(new_docs)

            if show_debug:
                logger.info(f"检索到 {len(new_docs)} 条文档")
                if len(state["docs"]) >= 2:
                    logger.info(
                        f"部分示例（共{len(state['docs'])}条）：\n\n{state['docs'][0].page_content[:200]}...\n\n{state['docs'][1].page_content[:200]}..."
                    )
                elif len(state["docs"]) == 1:
                    logger.info(f"仅检索一条数据：\n\n{state['docs'][0].page_content[:200]}")
                else:
                    logger.warning("未检索到任何文档！")
        except Exception as e:
            logger.error(f"检索过程出错: {e}")

    return state


def create_web_search_tool(
        search_cnt: int = 10,
        power_model: BaseChatModel = None
):
    """
    创建网络搜索工具

    Args:
        search_cnt: 网络搜索返回结果数量
        power_model: LLM实例

    Returns:
        tuple: (web_search_tool, web_search_llm)
    """
    if power_model is None:
        return None, None

    # 提前获取 WebSearcher 实例
    web_searcher = get_ws()

    @tool("web_search")
    def web_search(query: str) -> str:
        """
        使用网络搜索引擎进行检索
        Args: query: 检索查询文本
        Returns: 检索结果的JSON字符串
        """
        results: List[Document] = web_searcher.search(query, search_cnt)
        # 转换Document对象为字典列表
        results_dict = []
        for doc in results:
            results_dict.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        # 将字典列表转换为JSON字符串
        import json
        json_str = json.dumps(results_dict, ensure_ascii=False)
        logger.info(f"返回JSON字符串长度: {len(json_str)}")

        return json_str

    web_search_tool = web_search
    web_search_llm = power_model.bind_tools([web_search_tool])

    return web_search_tool, web_search_llm


# =============================================================================
# 主函数：功能测试
# =============================================================================
if __name__ == "__main__":
    import os
    from langchain_openai import ChatOpenAI

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("web_search.py 功能测试")
    print("=" * 60)

    # 配置参数（不再使用配置文件）
    network_search_cnt = 10  # 网络搜索返回结果数量

    # 1. 测试初始化
    print("\n[测试1] 初始化组件...")
    try:
        # 先初始化 WebSearcher
        reset_kb()  # 重置单例
        get_ws({})  # 初始化（WebSearcher 当前不需要配置）
        print("✓ WebSearcher 初始化成功")

        # 创建LLM实例（需要配置API Key）
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "qwen3:4b"),
            temperature=0.1,
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "sk-xxx")
        )
        print(f"✓ LLM初始化成功")

    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # 2. 测试创建网络搜索工具
    print("\n[测试2] 创建网络搜索工具...")
    search_tool = None
    search_llm = None
    try:
        # 创建网络搜索工具
        search_tool, search_llm = create_web_search_tool(
            search_cnt=network_search_cnt,
            power_model=llm
        )
        print(f"✓ 网络搜索工具创建成功")

    except Exception as e:
        print(f"✗ 创建网络搜索工具失败: {e}")
        import traceback

        traceback.print_exc()

    # 3. 测试网络搜索执行
    print("\n[测试3] 执行网络搜索...")
    if llm and search_tool:
        try:
            search_query = "阿司匹林副作用"
            print(f"  搜索查询: {search_query}")

            # 直接调用搜索工具
            result = search_tool.invoke(search_query)
            print(f"✓ 网络搜索执行成功")
            print(f"  - 结果长度: {len(result)} 字符")

            # 解析并显示结果
            import json

            results_data = json.loads(result)
            print(f"  - 检索到 {len(results_data)} 条结果")
            if results_data:
                print(f"  - 第一条结果预览: {results_data[0]['page_content'][:1000]}...")

        except Exception as e:
            print(f"✗ 网络搜索执行失败: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠ 前置条件不满足，跳过搜索执行测试")

    # 4. 测试llm_network_search节点（需要模拟state）
    print("\n[测试4] 测试llm_network_search节点...")
    if llm and search_tool:
        try:
            from langchain_core.documents import Document

            # 模拟输入状态
            test_state: WebSearchState = {
                "query": "阿司匹林有哪些副作用？",
                "other_messages": [],
                "docs": [
                    Document(
                        page_content="阿司匹林是一种非甾体抗炎药，主要用于解热镇痛。",
                        metadata={"source": "medical_db_1"}
                    )
                ]
            }

            print(f"  查询问题: {test_state['query']}")
            print(f"  输入文档数: {len(test_state['docs'])}")

            # 执行网络搜索节点
            result_state = llm_network_search(
                state=test_state,
                llm=search_llm,
                search_tool=search_tool,
                show_debug=True
            )

            print(f"✓ llm_network_search节点执行成功")
            print(f"  - 输出文档数: {len(result_state.get('docs', []))}")
            print(f"  - 其他消息数: {len(result_state.get('other_messages', []))}")
            if len(result_state.get('docs', [])) > 0:
                print(f"  - 第一个文档内容: {result_state.get('docs', [])[0].page_content[:100]}")

        except Exception as e:
            print(f"✗ llm_network_search节点执行失败: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠ 前置条件不满足，跳过节点测试")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
