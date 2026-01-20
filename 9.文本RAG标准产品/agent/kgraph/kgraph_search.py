"""
知识图谱检索模块
参考milvus模块的实现，提供图谱检索工具
"""
import json
import logging
import sys
from pathlib import Path
from typing import List
from typing import TYPE_CHECKING

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode

# 添加当前模块目录到 Python 路径（支持直接运行）
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 导入配置和工具函数
try:
    # 尝试相对导入（当作为包导入时）
    from ...prompts.templates import get_prompt_template
    from ...config.models import AppConfig
    from ..utils import json_to_list_document, _should_call_tool
    from .neo4j_connection import Neo4jConnection
    from .graph_searcher import GraphSearcher
except ImportError:
    # 回退到直接导入（当直接运行文件时）
    from prompts.templates import get_prompt_template
    from config.models import AppConfig
    from agent.utils import json_to_list_document, _should_call_tool
    from kgraph.neo4j_connection import Neo4jConnection
    from kgraph.graph_searcher import GraphSearcher

if TYPE_CHECKING:
    from typing_extensions import TypedDict


    class SearchMessagesState(TypedDict, total=False):
        query: str
        main_messages: List
        other_messages: List
        docs: List[Document]
        answer: str
        retry: int
        final: str
        judge_result: str

logger = logging.getLogger(__name__)


def llm_kgraph_search(
        state: "SearchMessagesState",
        llm: BaseChatModel,
        kgraph_tool_node: ToolNode,
        show_debug: bool
) -> "SearchMessagesState":
    """
    知识图谱检索节点

    ========== 功能说明 ==========
    该节点负责：
    1. 接收用户查询，让LLM判断是否需要调用知识图谱检索工具
    2. 如果需要，执行图谱检索并获取相关实体和关系
    3. 将检索到的实体/关系转换为Document对象添加到状态中供后续RAG使用
    """
    query = state["query"]

    if show_debug:
        logger.info(f"开始图谱检索节点，查询: {query}")

    # 调用LLM，让其判断是否需要调用图谱检索工具
    kg_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("call_kgraph")["system"]),
        HumanMessage(content=get_prompt_template("call_kgraph")["user"].format(query=query))
    ])
    state["other_messages"].append(kg_ai)

    # 检查LLM是否决定调用工具
    if _should_call_tool(kg_ai):
        if show_debug:
            tool_calls = getattr(kg_ai, 'tool_calls', None)
            if tool_calls and len(tool_calls) > 0:
                try:
                    if hasattr(tool_calls[0], 'args'):
                        args = tool_calls[0].args
                    elif isinstance(tool_calls[0], dict):
                        args = tool_calls[0].get('args', {})
                    else:
                        args = {}
                    logger.info(f"开始图谱检索，检索参数：{args}")
                except Exception as e:
                    logger.error(f"获取工具参数失败: {e}")

        try:
            # 执行工具调用
            tool_msgs: ToolMessage = kgraph_tool_node.invoke([kg_ai])
            state["other_messages"].append(tool_msgs)

            # 将ToolMessage中的JSON字符串转换为Document对象列表
            new_docs = json_to_list_document(tool_msgs[0].content)
            state["docs"].extend(new_docs)

            if show_debug:
                logger.info(f"图谱检索到 {len(new_docs)} 条文档")
                if len(state["docs"]) >= 2:
                    logger.info(
                        f"部分示例（共{len(state['docs'])}条）：\n\n{state['docs'][0].page_content[:200]}...\n\n{state['docs'][1].page_content[:200]}..."
                    )
                elif len(state["docs"]) == 1:
                    logger.info(f"仅检索一条数据：\n\n{state['docs'][0].page_content[:200]}")
                else:
                    logger.warning("未检索到任何图谱信息！")
        except Exception as e:
            logger.error(f"图谱检索过程出错: {e}")

    return state


def create_kgraph_search_tool(
        config,
        power_model: BaseChatModel
):
    """
    创建知识图谱检索工具节点

    Args:
        config: 应用配置
        power_model: LLM实例

    Returns:
        tuple: (kgraph_search_tool, kgraph_search_llm, kgraph_tool_node)
    """
    if config.agent.kgraph_search_enabled is False:
        return None, None, None

    cnt = config.agent.kgraph_search_cnt

    # 创建Neo4j连接
    neo4j_conn = Neo4jConnection(config)
    connected = neo4j_conn.connect()

    if not connected:
        logger.warning(f"Neo4j连接失败: {neo4j_conn.uri}")
        return None, None, None

    # 创建图谱检索器
    graph_searcher = GraphSearcher(neo4j_conn)

    @tool("kgraph_search")
    def kgraph_search(query: str) -> str:
        """
        知识图谱检索工具

        Args:
            query: 检索查询文本

        Returns:
            检索结果的JSON字符串
        """
        results = graph_searcher.search_graph(query, limit=cnt)
        # 转换Document对象为字典列表
        results_dict = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
        return json.dumps(results_dict, ensure_ascii=False)

    kgraph_search_tool = kgraph_search
    kgraph_search_llm = power_model.bind_tools([kgraph_search_tool])
    kgraph_tool_node = ToolNode([kgraph_search_tool])

    return kgraph_search_tool, kgraph_search_llm, kgraph_tool_node


# 使用示例
if __name__ == "__main__":

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("知识图谱检索示例")
    print("=" * 60)

    try:
        # 加载配置
        from config.loader import load_config

        config = load_config()

        print(f"\n📊 配置信息:")
        print(f"   Neo4j URI: {config.neo4j.uri}")
        print(f"   数据库: {config.neo4j.database}")

        # 创建Neo4j连接
        print(f"\n🔌 连接Neo4j数据库...")
        neo4j_conn = Neo4jConnection(config)
        connected = neo4j_conn.connect()

        if not connected:
            print(f"❌ Neo4j连接失败: {neo4j_conn.uri}")
            print("   请确保Neo4j服务已启动，并检查配置文件中的连接信息")
            exit(1)

        print(f"✅ Neo4j连接成功")

        # 创建图谱检索器
        graph_searcher = GraphSearcher(neo4j_conn)

        # 示例1: 关键词检索
        print(f"\n" + "=" * 60)
        print("示例1: 关键词检索")
        print("=" * 60)
        keyword = "感冒"
        print(f"搜索关键词: '{keyword}'")
        docs = graph_searcher.search_by_keyword(keyword, limit=5)
        print(f"✅ 找到 {len(docs)} 个实体:")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc.page_content}")

        # 示例2: 关系检索
        print(f"\n" + "=" * 60)
        print("示例2: 关系检索")
        print("=" * 60)
        entity_name = "阿司匹林"
        print(f"查询实体: '{entity_name}' 的关系")
        docs = graph_searcher.search_by_relation(entity_name, limit=5)
        print(f"✅ 找到 {len(docs)} 条关系:")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc.page_content}")

        # 示例3: 综合图谱检索
        print(f"\n" + "=" * 60)
        print("示例3: 综合图谱检索")
        print("=" * 60)
        keyword = "糖尿病"
        print(f"综合检索关键词: '{keyword}'")
        docs = graph_searcher.search_graph(keyword, limit=10)
        print(f"✅ 找到 {len(docs)} 条结果（实体+关系）:")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc.page_content}")

        # 示例4: 测试检索工具
        print(f"\n" + "=" * 60)
        print("示例4: 创建检索工具并调用")
        print("=" * 60)

        # 初始化LLM
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config.model.llm_model,
            temperature=config.llm.temperature,
            base_url=config.llm.base_url,
            api_key=config.llm.api_key or "dummy-key"
        )

        # 创建检索工具
        kgraph_tool, kgraph_llm, kgraph_tool_node = create_kgraph_search_tool(config, llm)

        if kgraph_tool is None:
            print("⚠️  图谱检索工具未启用")
        else:
            print(f"✅ 图谱检索工具创建成功")
            print(f"   工具名称: {kgraph_tool.name}")
            print(f"   工具描述: {kgraph_tool.description}")

            # 执行工具调用
            print(f"\n🔍 使用工具搜索: '高血压'")
            from langchain_core.messages import HumanMessage

            result = kgraph_tool.invoke({"query": "高血压"})
            print(f"✅ 检索结果（前500字符）:")
            print(f"   {str(result)[:500]}...")

        # 关闭连接
        neo4j_conn.close()
        print(f"\n✅ 连接已关闭")

        print(f"\n" + "=" * 60)
        print("图谱检索示例完成")
        print("=" * 60)

    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("   请确保已安装所需依赖: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        import traceback

        traceback.print_exc()
