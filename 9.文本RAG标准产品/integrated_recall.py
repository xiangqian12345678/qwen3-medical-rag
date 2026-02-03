"""搜索图：用于执行单个查询的RAG检索流程"""
import logging
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from typing_extensions import TypedDict

from app_config import APPConfig
from enhance.agent_state import AgentState
from recall.kgraph import create_kgraph_search_tool
from recall.kgraph.kg_utils import json_to_list_document as kg_json_to_list_document
from recall.milvus import create_db_search_tool
from recall.milvus.embed_utils import json_to_list_document
from recall.search import create_web_search_tool
from recall.search.search_utils import json_to_list_document as web_json_to_list_document

logger = logging.getLogger(__name__)


# =============================================================================
# 搜索图状态定义
# =============================================================================
class RecallState(TypedDict, total=False):
    query: str  # 用户查询
    other_messages: List  # 其他消息
    docs: List[Document]  # 检索到的文档


# =============================================================================
# 集成搜索
# =============================================================================
class IntegratedRecall:
    """搜索图：执行单个查询的RAG检索流程"""

    def __init__(self, app_config: APPConfig, llm: BaseChatModel, embed_model: Embeddings = None) -> None:
        """
        初始化搜索图

        Args:
            app_config: 应用配置
            llm: 强大模型实例
            websearch_func: 网络搜索函数
        """
        self.appConfig = app_config

        # 1.创建数据库检索工具
        db_tool, db_llm = create_db_search_tool(app_config.milvus_config_loader, llm, embed_model=embed_model)
        self.db_search_tool = db_tool
        self.db_search_llm = db_llm

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
            kgraph_tool, kgraph_llm \
                = create_kgraph_search_tool(app_config.kgraph_config_loader, llm, embed_model=embed_model)
            self.kgraph_search_tool = kgraph_tool
            self.kgraph_search_llm = kgraph_llm
        else:
            self.kgraph_search_tool = None
            self.kgraph_search_llm = None

    def search(self, query: str, agent_state: AgentState = None) -> List[Document]:
        docs: List[Document] = []

        from utils import invoke_with_timing

        # 1. 调用数据库检索
        def _db_search():
            return self.db_search_tool.invoke({"query": query})

        tool_result = invoke_with_timing(
            func=_db_search,
            stage_name="db_search",
            state=agent_state
        )
        db_docs = json_to_list_document(tool_result)
        docs.extend(db_docs)

        # 2. 调用网络检索
        if self.appConfig.agent_config.network_search_enabled and self.network_search_tool is not None:
            def _web_search():
                return self.network_search_tool.invoke(query)

            tool_result = invoke_with_timing(
                func=_web_search,
                stage_name="web_search",
                state=agent_state
            )
            web_docs = web_json_to_list_document(tool_result)
            docs.extend(web_docs)

        # 3. 调用知识图谱检索
        if self.appConfig.agent_config.kgraph_search_enabled and self.kgraph_search_tool is not None:
            def _kgraph_search():
                return self.kgraph_search_tool.invoke(query)

            tool_result = invoke_with_timing(
                func=_kgraph_search,
                stage_name="kgraph_search",
                state=agent_state
            )
            kg_docs = kg_json_to_list_document(tool_result)
            docs.extend(kg_docs)

        return docs

    def search_parallel(self, query: str, agent_state: AgentState = None) -> List[Document]:
        """
        并行检索函数: 同时执行数据库检索、网络检索和知识图谱检索

        Args:
            query: 检索查询
            agent_state: Agent状态对象,用于记录耗时信息

        Returns:
            检索到的文档列表
        """
        import concurrent.futures
        from utils import invoke_with_timing

        docs: List[Document] = []

        # 定义检索任务列表
        search_tasks = []

        # 1. 数据库检索任务(始终执行)
        def _db_search():
            tool_result = invoke_with_timing(
                func=lambda: self.db_search_tool.invoke({"query": query}),
                stage_name="db_search",
                state=agent_state
            )
            return json_to_list_document(tool_result)

        search_tasks.append(("db", _db_search))

        # 2. 网络检索任务(根据配置决定是否执行)
        if self.appConfig.agent_config.network_search_enabled and self.network_search_tool is not None:
            def _web_search():
                tool_result = invoke_with_timing(
                    func=lambda: self.network_search_tool.invoke(query),
                    stage_name="web_search",
                    state=agent_state
                )
                return web_json_to_list_document(tool_result)

            search_tasks.append(("web", _web_search))

        # 3. 知识图谱检索任务(根据配置决定是否执行)
        if self.appConfig.agent_config.kgraph_search_enabled and self.kgraph_search_tool is not None:
            def _kgraph_search():
                tool_result = invoke_with_timing(
                    func=lambda: self.kgraph_search_tool.invoke(query),
                    stage_name="kgraph_search",
                    state=agent_state
                )
                return kg_json_to_list_document(tool_result)

            search_tasks.append(("kgraph", _kgraph_search))

        # 并行执行所有检索任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(search_tasks)) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(task_func): task_name
                for task_name, task_func in search_tasks
            }

            # 等待所有任务完成并收集结果
            for future in concurrent.futures.as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result_docs = future.result()
                    if result_docs:
                        docs.extend(result_docs)
                        logger.info(f"{task_name}检索完成,返回{len(result_docs)}个文档")
                except Exception as e:
                    logger.error(f"{task_name}检索失败: {e}")

        return docs
