"""搜索图：用于执行单个查询的RAG检索流程"""
import logging
import sys
from functools import partial
from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv

from app_config import APPConfig
from recall.kgraph.kg_loader import KGraphConfigLoader
from recall.milvus.embed_loader import EmbedConfigLoader

# 加载环境变量（优先级：.env文件 > 系统环境变量）
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# 尝试相对导入（当作为包导入时）
from enhance.utils import create_llm_client

# 尝试相对导入（当作为包导入时）
from recall.milvus import llm_db_search, create_db_search_tool
from recall.search import llm_network_search, create_web_search_tool
from recall.kgraph import llm_kgraph_search, create_kgraph_search_tool
from rag_answer import rag_node, judge_node, finish_success, finish_fail


# =============================================================================
# 搜索图状态定义
# =============================================================================
class SearchMessagesState(TypedDict, total=False):
    """
    搜索消息状态

    用于在RAG检索流程中传递和管理状态信息。该状态类定义了整个搜索流程中
    所需的数据结构和流转信息，包括用户查询、对话历史、检索文档、生成答案等。

    ========== 使用案例 ==========
    场景：用户问"阿司匹林有哪些副作用？"

    初始状态：
    ```python
    init_state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [HumanMessage(content="阿司匹林有哪些副作用？")],
        "other_messages": [],
        "docs": [],
        "answer": "",
        "retry": 3,  # 最多重试3次
        "final": "",
        "judge_result": ""
    }
    ```

    流程演进：

    1. 数据库检索后（llm_db_search）：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [HumanMessage(content="阿司匹林有哪些副作用？")],
        "other_messages": [AIMessage(...), ToolMessage(...)],  # LLM调用和工具返回
        "docs": [
            Document(page_content="阿司匹林常见副作用包括：胃肠道反应、出血倾向、过敏反应等..."),
            Document(page_content="阿司匹林严重副作用：消化道溃疡、颅内出血等...")
        ],
        "answer": "",
        "retry": 3,
        "final": "",
        "judge_result": ""
    }
    ```

    2. RAG生成后（rag_node）：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [
            HumanMessage(content="阿司匹林有哪些副作用？"),
            AIMessage(content="根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...")
        ],
        "other_messages": [AIMessage(...), ToolMessage(...)],
        "docs": [Document(...), Document(...)],
        "answer": "根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...",  # RAG生成的答案
        "retry": 3,
        "final": "",
        "judge_result": ""
    }
    ```

    3. 质量判断后（judge_node）：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [...],
        "other_messages": [AIMessage(...), ToolMessage(...), AIMessage(content="[JUDGE]=yes")],
        "docs": [Document(...), Document(...)],
        "answer": "根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...",
        "retry": 3,
        "final": "",
        "judge_result": "pass"  # 质量判断通过
    }
    ```

    4. 成功结束后（finish_success）：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [...],
        "other_messages": [...],
        "docs": [Document(...), Document(...)],
        "answer": "根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...",
        "retry": 3,
        "final": "根据医学资料，阿司匹林的副作用包括：1.胃肠道反应...",  # 最终答案
        "judge_result": "pass"
    }
    ```

    重试场景示例：

    如果judge判断为retry：
    ```python
    state = {
        "query": "阿司匹林有哪些副作用？",
        "main_messages": [...],
        "other_messages": [...],
        "docs": [Document(...), Document(...)],  # 可能会替换或补充文档
        "answer": "答案不完整...",
        "retry": 2,  # 剩余2次重试机会（初始3次-1）
        "final": "",
        "judge_result": "retry"
    }
    ```

    ========== 字段详解 ==========

    Attributes:
        query: 用户查询
            - 类型: str
            - 作用: 存储原始用户问题，贯穿整个搜索流程
            - 示例: "阿司匹林有哪些副作用？"

        main_messages: 主要对话消息
            - 类型: List[Union[HumanMessage, AIMessage]]
            - 作用: 存储用户和AI的主要对话历史，用于生成最终回答
            - 包含: HumanMessage（用户消息）和AIMessage（AI回答）
            - 示例: [HumanMessage(content="问题"), AIMessage(content="回答")]

        other_messages: 其他消息
            - 类型: List[Union[SystemMessage, ToolMessage]]
            - 作用: 存储系统消息和工具调用消息，用于调试和追踪
            - 包含: SystemMessage（系统提示）、ToolMessage（工具返回）
            - 示例: [AIMessage(content="工具调用决策"), ToolMessage(content="检索结果")]

        docs: 检索到的文档
            - 类型: List[Document]
            - 作用: 存储从数据库/网络检索到的相关文档
            - 结构: Document(page_content="内容", metadata={...})
            - 示例: [Document(page_content="药物说明..."), Document(page_content="临床数据...")]

        answer: RAG生成的回答
            - 类型: str
            - 作用: 存储RAG模块生成的原始答案（可能需要质量评估）
            - 示例: "根据检索到的文档，该药物的副作用包括..."

        retry: 剩余重试次数
            - 类型: int
            - 作用: 在答案质量不合格时，允许重新生成的次数
            - 默认值: 通常为3次
            - 变化: 每次重试减1，减至0时停止重试

        final: 最终回答
            - 类型: str
            - 作用: 经过质量评估后的最终答案，可直接返回给用户
            - 特点: 质量不达标时会附加警告信息
            - 示例: "根据医学资料..." 或 "...（内容可能不属实）"

        judge_result: RAG质量评判结果
            - 类型: str
            - 作用: 存储质量评估结果，用于路由决策
            - 可能值:
                - "pass": 质量达标，流程结束
                - "retry": 质量不达标，重新生成（retry > 0）
                - "fail": 质量不达标且无重试机会，流程结束
            - 示例: "pass" / "retry" / "fail"

    ========== 字段间关系 ==========

    数据流向：
    query -> db_search/web_search -> docs -> rag -> answer -> judge -> judge_result -> final

    消息分类：
    - main_messages: 生成最终答案所需的对话流
    - other_messages: 辅助信息，用于调试和日志记录

    重试机制：
    retry初始值（如3）-> 判断为retry时减1 -> 为0时转为fail

    ========== total=False 说明 ==========

    total=False表示该TypedDict的所有字段都是可选的，不强制要求每次都提供所有字段。
    这样允许在不同节点只更新需要修改的字段，而不需要每次都传递完整状态。
    """
    query: str  # 用户查询
    main_messages: List[Union[HumanMessage, AIMessage]]  # 主要对话消息
    other_messages: List  # 其他消息
    docs: List[Document]  # 检索到的文档
    answer: str  # RAG生成的回答
    retry: int  # 剩余重试次数
    final: str  # 最终回答
    judge_result: str  # RAG质量评判结果


# =============================================================================
# 搜索图类
# =============================================================================
class SearchGraph:
    """搜索图：执行单个查询的RAG检索流程"""

    def __init__(self, appConfig: APPConfig, power_model: BaseChatModel) -> None:
        """
        初始化搜索图

        Args:
            appConfig: 应用配置
            power_model: 强大模型实例（用于工具调用）
            websearch_func: 网络搜索函数（可选）
        """
        self.appConfig = appConfig

        # 创建数据库检索工具
        db_tool, db_llm, db_node = create_db_search_tool(appConfig.milvus_config_loader, power_model)
        self.db_search_tool = db_tool
        self.db_search_llm = db_llm
        self.db_tool_node = db_node

        # 创建网络搜索工具(如果未启动，均赋值为: None)
        if appConfig.agent_config.network_search_enabled:
            web_tool, web_llm = create_web_search_tool(
                search_cnt=appConfig.agent_config.network_search_cnt,
                power_model=power_model
            )
            self.network_search_tool = web_tool
            self.network_search_llm = web_llm
        else:
            self.network_search_tool = None
            self.network_search_llm = None

        # 创建知识图谱检索工具
        # self.kgraphConfigLoader = KGraphConfigLoader()

        kgraph_tool, kgraph_llm, kgraph_node = create_kgraph_search_tool(appConfig.kgraph_config_loader, power_model)
        self.kgraph_search_tool = kgraph_tool
        self.kgraph_search_llm = kgraph_llm
        self.kgraph_tool_node = kgraph_node

        # 创建用于回答的LLM客户端
        self.llm = create_llm_client(appConfig.llm_config)
        self.search_graph = None

    def build_search_graph(self):
        """构建搜索图"""

        def judge_router(state: SearchMessagesState) -> str:
            """路由函数"""
            return state.get("judge_result", "fail")

        graph = StateGraph(SearchMessagesState)

        db_search_node_func = partial(
            llm_db_search,
            llm=self.db_search_llm,
            db_tool_node=self.db_tool_node,
            show_debug=self.appConfig.dialogue_config.console_debug
        )
        graph.add_node("db_search", db_search_node_func)

        # 添加web_search节点
        if self.appConfig.agent_config.network_search_enabled and self.network_search_tool is not None:
            network_search_node_func = partial(
                llm_network_search,
                llm=self.network_search_llm,
                search_tool=self.network_search_tool,
                show_debug=self.appConfig.dialogue_config.console_debug
            )
            graph.add_node("web_search", network_search_node_func)

        # 添加知识图谱搜索节点
        if self.appConfig.agent_config.kgraph_search_enabled and self.kgraph_tool_node is not None:
            kgraph_search_node_func = partial(
                llm_kgraph_search,
                llm=self.kgraph_search_llm,
                kgraph_tool_node=self.kgraph_tool_node,
                show_debug=self.appConfig.dialogue_config.console_debug
            )
            graph.add_node("kgraph_search", kgraph_search_node_func)

        # 添加RAG节点
        rag_node_func = partial(
            rag_node,
            llm=self.llm,
            show_debug=self.appConfig.dialogue_config.console_debug
        )
        graph.add_node("rag", rag_node_func)

        # 1.3 添加结束节点
        graph.add_node("finish_success", finish_success)
        graph.add_node("finish_fail", finish_fail)

        # 1.4 添加判断节点
        judge_node_func = partial(
            judge_node,
            llm=self.llm,
            show_debug=self.appConfig.dialogue_config.console_debug
        )
        graph.add_node("judge", judge_node_func)

        # 构建检索流程：db_search -> web_search(可选) -> kgraph_search(可选) -> rag
        # 2.设置边
        # 2.1 召回流程
        # 2.1.1 向量召回为入口节点
        graph.set_entry_point("db_search")
        last_node = "db_search"

        # 2.1.2 网络召回
        if self.appConfig.agent_config.network_search_enabled and self.network_search_tool is not None:
            graph.add_edge("db_search", "web_search")
            last_node = "web_search"

        # 2.1.3 知识图谱召回
        if self.appConfig.agent_config.kgraph_search_enabled and self.kgraph_tool_node is not None:
            graph.add_edge(last_node, "kgraph_search")
            last_node = "kgraph_search"

        graph.add_edge(last_node, "rag")
        graph.add_edge("rag", "judge")
        graph.add_conditional_edges(
            "judge",
            judge_router,
            {
                "pass": "finish_success",
                "retry": "rag",
                "fail": "finish_fail"
            }
        )
        graph.add_edge("finish_success", END)
        graph.add_edge("finish_fail", END)

        self.search_graph = graph.compile()

    def run(self, init_state: SearchMessagesState) -> SearchMessagesState:
        """
        执行搜索图
        Args:    init_state: 初始状态
        Returns: 执行完成后的最终状态
        """
        if self.search_graph is None:
            self.build_search_graph()

        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state

    def answer(self, query: str) -> str:
        """
        回答用户问题

        Args:
            query: 用户问题

        Returns:
            回答结果
        """
        if self.search_graph is None:
            self.build_search_graph()

        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "answer": "",
            "retry": self.appConfig.agent_config.max_attempts,
            "final": "",
            "judge_result": ""
        }

        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state.get("final", "") or out_state.get("answer", "") or "（空）"


# =============================================================================
# Main 函数
# =============================================================================
if __name__ == "__main__":
    """
    单代码执行样例

    ============ 使用场景说明 ============

    SearchGraph 是执行单个查询的RAG检索流程，适用于以下场景：
    1. 独立的医疗问答：针对单个问题进行精确检索和回答
    2. 多轮对话中的子查询：作为MultiDialogueAgent的底层组件处理拆分后的子查询
    3. 快速验证：测试RAG系统和向量数据库的检索效果

    ============ 运行前准备 ============

    1. 确保 Milvus 向量数据库已启动并包含医疗知识数据
       docker ps  # 查看milvus容器是否运行

    2. 确保 Ollama 服务已启动并下载模型
       ollama serve
       ollama pull bge-m3:latest  # 嵌入模型
       ollama pull qwen3:4b      # LLM模型

    3. 确保配置文件路径正确（当前目录下的 rag_config.yaml）

    ============ 运行方式 ============

    方式1：直接执行当前文件
    python agent/search_graph.py

    方式2：在其他脚本中导入使用
    from agent.search_graph import SearchGraph
    from config.loader import ConfigLoader
    from core.utils import create_llm_client

    config = ConfigLoader().config
    power_model = create_llm_client(config.llm)
    search_graph = SearchGraph(config, power_model=power_model)
    result = search_graph.answer("阿司匹林有哪些副作用？")
    print(result)
    """

    import logging
    from rag_loader import RAGConfigLoader

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 60)
    logger.info("SearchGraph 单查询RAG系统")
    logger.info("=" * 60)

    try:
        # 1. 加载配置
        logger.info("\n[1/5] 加载配置...")
        rag_config_loader = RAGConfigLoader()  # 与milvus_config 和 kgraph_config 的loader不同
        milvus_config_loader = EmbedConfigLoader()
        kgraph_config_loader = KGraphConfigLoader()
        app_config = APPConfig(rag_config_loader=rag_config_loader,
                               milvus_config_loader=milvus_config_loader,
                               kgraph_config_loader=kgraph_config_loader)
        rag_config = rag_config_loader.config

        logger.info(f"配置加载成功!")
        logger.info(f"  - Milvus: {milvus_config_loader.milvus.uri}")
        logger.info(f"  - Collection: {milvus_config_loader.milvus.collection_name}")
        logger.info(f"  - LLM: {rag_config.llm_config.model} ({rag_config.llm_config.provider})")
        logger.info(f"  - 网络搜索: {'启用' if rag_config.agent_config.network_search_enabled else '禁用'}")

        # 2. 创建LLM客户端
        logger.info("\n[2/5] 创建LLM客户端...")
        power_model = create_llm_client(rag_config.llm_config)
        logger.info(f"LLM客户端创建成功!")

        # 3. 初始化SearchGraph
        logger.info("\n[3/5] 初始化SearchGraph...")
        search_graph = SearchGraph(app_config, power_model=power_model)
        search_graph.build_search_graph()
        logger.info("SearchGraph初始化成功!")
        logger.info(f"图结构: db_search -> web_search(可选) -> 图谱搜索(可选) -> rag -> judge(可选) -> finish")

        # 4. 执行示例查询
        logger.info("\n[4/5] 执行示例查询...")

        # 示例查询列表（可根据需要修改或添加）
        example_queries = [
            "什么是高血压？",  # 触发向量检索+网络检索
            "房颤的治疗目的是什么？"  # 触发图谱检索+网络检索+图谱检索
        ]

        # 选择要执行的查询（修改索引选择不同查询）
        selected_query_index = 1
        query = example_queries[selected_query_index]

        logger.info(f"查询: {query}")
        logger.info("-" * 60)

        # 执行查询
        result = search_graph.answer(query)

        # 5. 输出结果
        logger.info("\n[5/5] 查询结果:")
        logger.info("=" * 60)
        print(f"\n问题: {query}\n")
        print(f"回答:\n{result}\n")
        logger.info("=" * 60)
        logger.info("查询完成!")

    except Exception as e:
        logger.error(f"\n执行失败: {e}", exc_info=True)
        sys.exit(1)
