"""搜索图：用于执行单个查询的RAG检索流程"""
import logging
from functools import partial
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from app_config import APPConfig

logger = logging.getLogger(__name__)

# 尝试相对导入（当作为包导入时）

# 尝试相对导入（当作为包导入时）
from recall.milvus import llm_db_search, create_db_search_tool
from recall.search import llm_network_search, create_web_search_tool
from recall.kgraph import llm_kgraph_search, create_kgraph_search_tool
from answer.answer import generate_answer, judge_node, finish_success, finish_fail


# =============================================================================
# 搜索图状态定义
# =============================================================================
class RecallState(TypedDict, total=False):
    query: str  # 用户查询
    other_messages: List  # 其他消息
    docs: List[Document]  # 检索到的文档


# =============================================================================
# 搜索工具
# =============================================================================
class RecallGraph:
    """搜索图：执行单个查询的RAG检索流程"""

    def __init__(self, app_config: APPConfig, llm: BaseChatModel, embed_model: Embeddings = None) -> None:
        """
        初始化搜索图

        Args:
            app_config: 应用配置
            llm: 强大模型实例（用于工具调用）
            websearch_func: 网络搜索函数（可选）
        """
        self.appConfig = app_config

        # 1.创建数据库检索工具
        db_tool, db_llm, db_node = create_db_search_tool(app_config.milvus_config_loader, llm, embed_model=embed_model)
        self.db_search_tool = db_tool
        self.db_search_llm = db_llm
        self.db_tool_node = db_node

        # 2.创建网络搜索工具
        if app_config.agent_config.network_search_enabled:
            web_tool, web_llm = create_web_search_tool(
                search_cnt=app_config.agent_config.network_search_cnt,
                power_model=llm
            )
            self.network_search_tool = web_tool
            self.network_search_llm = web_llm
        else:
            self.network_search_tool = None
            self.network_search_llm = None

        # 3.创建知识图谱搜索工具
        if app_config.agent_config.kgraph_search_enabled:
            kgraph_tool, kgraph_llm, kgraph_node = create_kgraph_search_tool(app_config.kgraph_config_loader, llm,
                                                                             embed_model=embed_model)
            self.kgraph_search_tool = kgraph_tool
            self.kgraph_search_llm = kgraph_llm
            self.kgraph_tool_node = kgraph_node
        else:
            self.kgraph_search_tool = None
            self.kgraph_search_llm = None
            self.kgraph_tool_node = None

        # 创建用于回答的LLM客户端
        self.llm = llm

        # 检索图
        self.search_graph = None

    def build_search_graph(self):
        def judge_router(state: RecallState) -> str:
            """路由函数"""
            return state.get("judge_result", "fail")

        # 1.创建检索图
        recall_graph = StateGraph(RecallState)

        # 2.创建节点
        # 2.1 创建db_search节点
        db_search_node_func = partial(
            llm_db_search,
            llm=self.db_search_llm,
            db_tool_node=self.db_tool_node,
            show_debug=self.appConfig.dialogue_config.console_debug
        )
        recall_graph.add_node("db_search", db_search_node_func)

        # 2.2 添加web_search节点
        if self.appConfig.agent_config.network_search_enabled and self.network_search_tool is not None:
            network_search_node_func = partial(
                llm_network_search,
                llm=self.network_search_llm,
                search_tool=self.network_search_tool,
                show_debug=self.appConfig.dialogue_config.console_debug
            )
            recall_graph.add_node("web_search", network_search_node_func)

        # 2.3 添加知识图谱搜索节点
        if self.appConfig.agent_config.kgraph_search_enabled and self.kgraph_tool_node is not None:
            kgraph_search_node_func = partial(
                llm_kgraph_search,
                llm=self.kgraph_search_llm,
                kgraph_tool_node=self.kgraph_tool_node,
                show_debug=self.appConfig.dialogue_config.console_debug
            )
            recall_graph.add_node("kgraph_search", kgraph_search_node_func)

        # 添加RAG节点
        answer_func = partial(
            generate_answer,
            llm=self.llm,
            show_debug=self.appConfig.dialogue_config.console_debug
        )
        recall_graph.add_node("rag", answer_func)

        # 1.3 添加结束节点
        recall_graph.add_node("finish_success", finish_success)
        recall_graph.add_node("finish_fail", finish_fail)

        # 1.4 添加判断节点
        judge_node_func = partial(
            judge_node,
            llm=self.llm,
            show_debug=self.appConfig.dialogue_config.console_debug
        )
        recall_graph.add_node("judge", judge_node_func)

        # 构建检索流程：db_search -> web_search(可选) -> kgraph_search(可选) -> rag
        # 2.设置边
        # 2.1 召回流程
        # 2.1.1 向量召回为入口节点
        recall_graph.set_entry_point("db_search")
        last_node = "db_search"

        # 2.1.2 网络召回
        if self.appConfig.agent_config.network_search_enabled and self.network_search_tool is not None:
            recall_graph.add_edge("db_search", "web_search")
            last_node = "web_search"

        # 2.1.3 知识图谱召回
        if self.appConfig.agent_config.kgraph_search_enabled and self.kgraph_tool_node is not None:
            recall_graph.add_edge(last_node, "kgraph_search")
            last_node = "kgraph_search"

        recall_graph.add_edge(last_node, "rag")
        recall_graph.add_edge("rag", "judge")
        recall_graph.add_conditional_edges(
            "judge",
            judge_router,
            {
                "pass": "finish_success",
                "retry": "rag",
                "fail": "finish_fail"
            }
        )
        recall_graph.add_edge("finish_success", END)
        recall_graph.add_edge("finish_fail", END)

        self.search_graph = recall_graph.compile()

    def run(self, init_state: RecallState) -> RecallState:
        """
        执行搜索图
        Args:    init_state: 初始状态
        Returns: 执行完成后的最终状态
        """
        if self.search_graph is None:
            self.build_search_graph()

        out_state: RecallState = self.search_graph.invoke(init_state)
        return out_state

    def search(self, query: str) -> List[Document]:
        from recall.milvus.embed_utils import json_to_list_document
        from recall.search.search_utils import json_to_list_document as web_json_to_list_document
        from recall.kgraph.kg_utils import json_to_list_document as kg_json_to_list_document

        docs: List[Document] = []
        # 1. 调用数据库检索
        tool_result = self.db_search_tool.invoke({"query": query})
        db_docs = json_to_list_document(tool_result)
        docs.extend(db_docs)

        # 2. 调用网络检索
        if self.appConfig.agent_config.network_search_enabled and self.network_search_tool is not None:
            tool_result = self.network_search_tool.invoke(query)
            web_docs = web_json_to_list_document(tool_result)
            docs.extend(web_docs)

        # 3. 调用知识图谱检索
        if self.appConfig.agent_config.kgraph_search_enabled and self.kgraph_tool_node is not None:
            tool_result = self.kgraph_search_tool.invoke(query)
            kg_docs = kg_json_to_list_document(tool_result)
            docs.extend(kg_docs)

        return docs

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

        init_state: RecallState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "answer": "",
            "retry": self.appConfig.agent_config.max_attempts,
            "final": "",
            "judge_result": ""
        }

        out_state: RecallState = self.search_graph.invoke(init_state)
        return out_state.get("final", "") or out_state.get("answer", "") or "（空）"
