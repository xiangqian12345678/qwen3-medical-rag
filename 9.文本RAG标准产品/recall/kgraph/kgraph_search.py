"""
知识图谱检索模块
参考milvus模块的实现，提供图谱检索工具
"""
import json
import logging
from typing import List
from typing_extensions import TypedDict

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from .kg_loader import KGraphConfigLoader
from .kg_templates import get_prompt_template
from .kg_utils import json_to_list_document, _should_call_tool
from .kgraph_searcher import GraphSearcher
from .neo4j_connection import Neo4jConnection


class KGraphRecallState(TypedDict, total=False):
    query: str
    other_messages: List
    docs: List[Document]


logger = logging.getLogger(__name__)


def llm_kgraph_search(
        state: "KGraphRecallState",
        llm: BaseChatModel,
        kgraph_tool_node: ToolNode,
        show_debug: bool
) -> "KGraphRecallState":
    """
    知识图谱检索节点

    ========== 功能说明 ==========
    该节点负责：
    1. 接收用户查询，让LLM判断是否需要调用知识图谱检索工具
    2. 如果需要，执行图谱检索并获取相关实体和关系
    3. 将检索到的实体/关系转换为Document对象添加到状态中供后续RAG使用
    """
    print('-' * 60)
    print("开始图谱检索")
    print('-' * 60)
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
        kgraph_config_loader: KGraphConfigLoader,
        power_model: BaseChatModel
):
    """
    创建知识图谱检索工具节点

    Args:
        kgraph_config_loader: 图谱配置加载器
        power_model: LLM实例

    Returns:
        tuple: (kgraph_search_tool, kgraph_search_llm, kgraph_tool_node)
    """
    # 默认启用知识图谱搜索
    cnt = 10  # 默认检索10条结果
    neo4j_conn = Neo4jConnection(kgraph_config_loader)  # 创建Neo4j连接
    connected = neo4j_conn.connect()

    if not connected:
        logger.warning(f"Neo4j连接失败: {neo4j_conn.uri}")
        return None, None, None

    # 创建图谱检索器（传入嵌入配置以支持向量检索）
    # 使用 text_dense 配置作为嵌入模型
    embedding_config = {
        "provider": kgraph_config_loader.get("embedding.provider", "ollama"),
        "model": kgraph_config_loader.get("embedding.model", "nomic-embed-text"),
        "api_key": kgraph_config_loader.get("embedding.api_key", None),
        "base_url": kgraph_config_loader.get("embedding.base_url", "http://localhost:11434/v1")
    }
    graph_searcher = GraphSearcher(neo4j_conn, embedding_config=embedding_config)

    @tool("kgraph_search")
    def kgraph_search(query: str) -> str:
        """
        知识图谱检索工具

        Args:
            query: 检索查询文本

        Returns:
            检索结果的JSON字符串
        """
        # 使用向量检索
        results = graph_searcher.search_graph_by_query(query_text=query, top_k=cnt)
        vdb_results = results.get("vdb_results", [])

        # 转换为Document对象
        results_dict = [{"page_content": doc, "metadata": {"source": "knowledge_graph", "query": query}} for doc in vdb_results]
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
        from kg_loader import KGraphConfigLoader

        config = KGraphConfigLoader()

        print(f"\n📊 配置信息:")
        print(f"   Neo4j URI: {config.neo4j_config.uri}")
        print(f"   数据库: {config.neo4j_config.database}")

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
        embedding_config = {
            "provider": config.get("embedding.provider", "ollama"),
            "model": config.get("embedding.model", "nomic-embed-text"),
            "api_key": config.get("embedding.api_key", None),
            "base_url": config.get("embedding.base_url", "http://localhost:11434/v1")
        }
        graph_searcher = GraphSearcher(neo4j_conn, embedding_config=embedding_config)

        # 示例1: query检索
        print(f"\n" + "=" * 60)
        print("示例1: 关键词检索")
        print("=" * 60)
        keyword = "房颤的治疗目的是什么？"
        print(f"搜索关键词: '{keyword}'")
        dict = graph_searcher.search_graph_by_query(keyword, top_k=5)
        content = dict.get("content", "")
        print(f"  content: {content}")

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

        # 示例3: 关键词检索
        print(f"\n" + "=" * 60)
        print("示例3: 综合图谱检索")
        print("=" * 60)
        keyword = "糖尿病"
        print(f"综合检索关键词: '{keyword}'")
        result = graph_searcher.search_by_keyword(keyword, limit=10)
        print(f"✅ 找到 {len(result)} 条结果（实体）:")
        for i, doc in enumerate(result, 1):
            print(f"   {i}. {doc.page_content}")

        # 示例4: 测试检索工具
        print(f"\n" + "=" * 60)
        print("示例4: 创建检索工具并调用")
        print("=" * 60)

        # 初始化LLM
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config.llm_config.model,
            temperature=config.llm_config.temperature,
            base_url=config.llm_config.base_url,
            api_key=config.llm_config.api_key or "dummy-key"
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
