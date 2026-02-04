import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import List

from langchain_community.document_compressors import DashScopeRerank
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from answer.answer import generate_answer, AnswerState
from app_config import APPConfig
from enhance.agent_state import AgentState, AskMess
from enhance.agent_state import RewriteQuery, SubQueries, MultiQueries, SuperordinateQuery, HypotheticalAnswer
from enhance.filter_enhance import filter_low_correction_doc_embeddings, filter_low_correction_doc_llm, \
    filter_low_correction_content, filter_redundant_doc_embeddings
from enhance.query_enhance import query_refine, generate_summary, query_rewrite
from enhance.recall_enhance import generate_multi_queries, generate_superordinate_query, generate_hypothetical_answer, \
    generate_sub_queries
from enhance.sort_enhance import sort_docs_cross_encoder, sort_docs_by_loss_of_location
from integrated_recall import IntegratedRecall
from rag.rag_config import AgentConfig
from utils import invoke_with_timing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class DialogueAgent:
    def __init__(self, app_config: APPConfig, embeddings_model: Embeddings, llm: BaseChatModel,
                 reranker: DashScopeRerank) -> None:
        self.app_config = app_config
        self.agent_config = app_config.agent_config
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.reranker = reranker
        self.recall_graph = IntegratedRecall(app_config, llm, embed_model=embeddings_model)

        # 对话状态
        self.agent_state = AgentState(
            curr_input="",
            curr_ask_num=0,
            max_ask_num=3,
            dialogue_messages=[],
            asking_messages=[],
            background_info="",
            performance=[],
            ask_obj=AskMess(need_ask=False, questions=[]),
            multi_summary=[]
        )

        # 代理总图
        self._build_graph()

    def answer(self, query: str) -> AgentState:
        # 创建输入状态
        self.agent_state["curr_input"] = query

        self.agent_state = self.agent_graph.invoke(self.agent_state)
        return self.agent_state

    def _build_graph(self):
        """构建代理总图"""
        # 1.构建代理总图
        graph: StateGraph[AgentState] = StateGraph(AgentState)  # type: ignore[arg-type]

        # 2.构建节点
        # 判断是否需要追问
        graph.add_node("ask_user", partial(query_refine, llm=self.llm))
        # 总结对话背景信息
        graph.add_node("query_enhance", partial(_query_enhance, agent_config=self.agent_config, llm=self.llm))
        # 召回增强
        graph.add_node("recall_enhance", partial(_recall_enhance, agent_config=self.agent_config, llm=self.llm))
        # 文档召回
        graph.add_node("recall", partial(_recall, recall_graph=self.recall_graph, agent_config=self.agent_config))
        # 文档过滤
        graph.add_node("filter_enhance", partial(_filter_enhance, agent_config=self.agent_config,
                                                 embeddings_model=self.embeddings_model, llm=self.llm))
        # 文档排序
        graph.add_node("sort_enhance", partial(_sort_enhance, agent_config=self.agent_config, reranker=self.reranker))
        # 生成答案
        graph.add_node("answer", partial(_answer, llm=self.llm, show_debug=True))

        # 3.构建边
        # 起始边
        graph.add_edge(START, "ask_user")
        # 条件边：
        # 1.需要补充信息就进入END，让用户补充信息
        # 2.不需要补充信息就进入extract，总结对话背景信息
        graph.add_conditional_edges("ask_user", route_ask_again, {"ask": END, "pass": "query_enhance"})
        # 召回强化： 生成多类扩展召回的query
        graph.add_edge("query_enhance", "recall_enhance")
        # 召回增强
        graph.add_edge("recall_enhance", "recall")
        # 文档召回
        graph.add_edge("recall", "filter_enhance")
        # 文档过滤
        graph.add_edge("filter_enhance", "sort_enhance")
        # 文档排序
        graph.add_edge("sort_enhance", "answer")
        # 回答问题
        graph.add_edge("answer", END)

        # 4.构建代理总图
        self.agent_graph = graph.compile()


def _query_enhance(agent_state: AgentState, agent_config: AgentConfig, llm: BaseChatModel) -> AgentState:
    """并行化的query增强处理，同时执行summary和query_rewrite任务"""
    new_state = agent_state.copy()
    start_time = time.time()

    def _task_generate_summary(state: AgentState) -> None:
        result = generate_summary(state, llm)
        state["background_info"] = result["background_info"]

    def _task_query_rewrite(state: AgentState) -> None:
        if agent_config.query_rewrite_enabled:
            result = query_rewrite(state, llm)
            state["rewrite_query"] = result["rewrite_query"]

    # 创建任务列表，generate_summary始终执行，query_rewrite根据配置执行
    tasks = [_task_generate_summary]
    if agent_config.query_rewrite_enabled:
        tasks.append(_task_query_rewrite)

    # 并行执行所有启用的任务
    if tasks:
        with ThreadPoolExecutor(max_workers=min(len(tasks), 2)) as executor:
            futures = [executor.submit(task, new_state) for task in tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Query enhance error: {e}")

    elapsed_time = time.time() - start_time

    # 收集query增强信息
    query_info = {}

    # summary 总是执行
    query_info["background_info"] = new_state.get("background_info", "")

    # query_rewrite 根据配置执行
    if agent_config.query_rewrite_enabled:
        rewrite_query: RewriteQuery = new_state.get("rewrite_query", RewriteQuery(query=""))
        query_info["rewrite_query"] = rewrite_query.query

    perf_info = {
        "stage": "_query_enhance",
        "duration": elapsed_time,
        "timestamp": time.time(),
        "query_info": query_info
    }
    new_state["performance"].append(perf_info)

    return new_state


def _recall_enhance(agent_state: AgentState, agent_config: AgentConfig, llm: BaseChatModel) -> AgentState:
    """并行化的召回增强处理，同时执行多个查询生成任务"""
    logger.info('\n' + '-' * 100)

    new_state = agent_state.copy()
    start_time = time.time()

    def _task_multi_queries(state: AgentState) -> None:
        if agent_config.generate_multi_queries_enabled:
            result = generate_multi_queries(state, llm)
            state["multi_query"] = result["multi_query"]

    def _task_superordinate_query(state: AgentState) -> None:
        if agent_config.generate_superordinate_query_enabled:
            result = generate_superordinate_query(state, llm)
            state["superordinate_query"] = result["superordinate_query"]

    def _task_hypothetical_answer(state: AgentState) -> None:
        if agent_config.generate_hypothetical_answer_enabled:
            result = generate_hypothetical_answer(state, llm)
            state["hypothetical_answer"] = result["hypothetical_answer"]

    def _task_sub_queries(state: AgentState) -> None:
        if agent_config.generate_sub_queries_enabled:
            result = generate_sub_queries(state, llm)
            state["sub_query"] = result["sub_query"]

    # 创建任务列表，只包含启用的任务
    tasks = []
    if agent_config.generate_multi_queries_enabled:
        tasks.append(_task_multi_queries)
    if agent_config.generate_superordinate_query_enabled:
        tasks.append(_task_superordinate_query)
    if agent_config.generate_hypothetical_answer_enabled:
        tasks.append(_task_hypothetical_answer)
    if agent_config.generate_sub_queries_enabled:
        tasks.append(_task_sub_queries)

    # 并行执行所有启用的任务
    if tasks:
        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
            futures = [executor.submit(task, new_state) for task in tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Recall enhance error: {e}")

    elapsed_time = time.time() - start_time

    # 收集各类强化query信息
    query_info = {}

    if agent_config.generate_multi_queries_enabled:
        multi_queries: MultiQueries = new_state.get("multi_query", MultiQueries(queries=[]))
        query_info["multi_query"] = multi_queries.queries

    if agent_config.generate_sub_queries_enabled:
        sub_queries: SubQueries = new_state.get("sub_query", SubQueries(need_split=False, queries=[]))
        query_info["sub_query"] = sub_queries.queries

    if agent_config.generate_superordinate_query_enabled:
        superordinate_query: SuperordinateQuery = new_state.get("superordinate_query",
                                                                SuperordinateQuery(superordinate_query=""))
        query_info["superordinate_query"] = superordinate_query.superordinate_query

    if agent_config.generate_hypothetical_answer_enabled:
        hypothetical_answer: HypotheticalAnswer = new_state.get("hypothetical_answer",
                                                                HypotheticalAnswer(hypothetical_answer=""))
        query_info["hypothetical_answer"] = hypothetical_answer.hypothetical_answer

    perf_info = {
        "stage": "_recall_enhance",
        "duration": elapsed_time,
        "timestamp": time.time(),
        "query_info": query_info
    }
    new_state["performance"].append(perf_info)

    logger.info('\n' + '-' * 100)
    logger.info(f"recall_enhance query_info: {query_info}")
    return new_state


def _recall(agent_state: AgentState, recall_graph: "IntegratedRecall", agent_config: AgentConfig) -> AgentState:
    import os
    cpu_count = os.cpu_count()
    new_state = agent_state.copy()
    start_time = time.time()

    def is_empty(s: str | None) -> bool:
        return not s or not s.strip()

    def _run_one(single_query: str) -> List[Document]:
        docs = recall_graph.search_parallel(query=single_query, agent_state=agent_state)
        return docs

    def _parallel_recall(query_list: List[str], max_parallel: int = 1) -> List[List[Document]]:
        all_docs: List[List[Document]] = []
        with ThreadPoolExecutor(max_workers=min(len(query_list), max_parallel)) as recall_executor:
            future_list = []
            for single_query in query_list:
                future_list.append(recall_executor.submit(_run_one, single_query))

            for single_future in as_completed(future_list):
                try:
                    docs = single_future.result()
                    all_docs.append(docs)
                except Exception as e:
                    print('generated an exception: %s' % e)
        return all_docs

    def _process_sub_queries() -> None:
        sub_queries: SubQueries = new_state.get("sub_query", SubQueries(need_split=False, queries=[]))
        if (agent_config.generate_sub_queries_enabled
                and sub_queries.need_split
                and len(sub_queries.queries) > 0):
            doc_list_list = invoke_with_timing(
                func=_parallel_recall,
                query_list=sub_queries.queries,
                max_parallel=cpu_count,
                stage_name="sub_query_parallel_recall",
                state=agent_state
            )
            new_state["sub_query_results"] = doc_list_list

    def _process_rewrite_query() -> None:
        rewrite_query: RewriteQuery = new_state.get("rewrite_query", RewriteQuery(query=""))

        if agent_config.query_rewrite_enabled and not is_empty(rewrite_query.query):
            docs = invoke_with_timing(
                func=_run_one,
                single_query=rewrite_query.query,
                stage_name="rewrite_query_recall",
                state=agent_state
            )
            new_state["rewrite_query_docs"] = docs

    def _process_multi_queries() -> None:
        multi_queries: MultiQueries = new_state.get("multi_query", MultiQueries(queries=[]))

        if agent_config.generate_multi_queries_enabled and len(multi_queries.queries) > 0:
            doc_list_list = invoke_with_timing(
                func=_parallel_recall,
                query_list=multi_queries.queries,
                max_parallel=cpu_count,
                stage_name="multi_query_parallel_recall",
                state=agent_state
            )
            new_state["multi_query_docs"] = doc_list_list

    def _process_superordinate_query() -> None:
        superordinate_query: SuperordinateQuery = new_state.get("superordinate_query", SuperordinateQuery())

        if agent_config.generate_superordinate_query_enabled and not is_empty(superordinate_query.superordinate_query):
            docs = invoke_with_timing(
                func=_run_one,
                single_query=superordinate_query.superordinate_query,
                stage_name="superordinate_query_recall",
                state=agent_state
            )
            new_state["superordinate_query_docs"] = docs

    def _process_hypothetical_answer() -> None:
        hypothetical_answer: HypotheticalAnswer = new_state.get("hypothetical_answer", HypotheticalAnswer())

        if agent_config.generate_hypothetical_answer_enabled and not is_empty(hypothetical_answer.hypothetical_answer):
            docs = invoke_with_timing(
                func=_run_one,
                single_query=hypothetical_answer.hypothetical_answer,
                stage_name="hypothetical_answer_recall",
                state=agent_state
            )
            new_state["hypothetical_answer_docs"] = docs

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(_process_sub_queries),
            executor.submit(_process_rewrite_query),
            executor.submit(_process_multi_queries),
            executor.submit(_process_superordinate_query),
            executor.submit(_process_hypothetical_answer)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f'Recall error: {e}')

    elapsed_time = time.time() - start_time

    # 收集各类召回的文档数统计
    doc_count_info = {}

    if agent_config.generate_sub_queries_enabled:
        sub_query_results: List[List[Document]] = new_state.get("sub_query_results", [])
        sub_doc_counts = [len(docs) for docs in sub_query_results]
        doc_count_info["sub_query_doc_counts"] = sub_doc_counts
        doc_count_info["sub_query_total_docs"] = sum(sub_doc_counts)

    if agent_config.query_rewrite_enabled:
        rewrite_query_docs: List[Document] = new_state.get("rewrite_query_docs", [])
        doc_count_info["rewrite_query_doc_count"] = len(rewrite_query_docs)

    if agent_config.generate_multi_queries_enabled:
        multi_query_docs: List[List[Document]] = new_state.get("multi_query_docs", [])
        multi_doc_counts = [len(docs) for docs in multi_query_docs]
        doc_count_info["multi_query_doc_counts"] = multi_doc_counts
        doc_count_info["multi_query_total_docs"] = sum(multi_doc_counts)

    if agent_config.generate_superordinate_query_enabled:
        superordinate_query_docs: List[Document] = new_state.get("superordinate_query_docs", [])
        doc_count_info["superordinate_query_doc_count"] = len(superordinate_query_docs)

    if agent_config.generate_hypothetical_answer_enabled:
        hypothetical_answer_docs: List[Document] = new_state.get("hypothetical_answer_docs", [])
        doc_count_info["hypothetical_answer_doc_count"] = len(hypothetical_answer_docs)

    # 计算总文档数
    total_docs = doc_count_info.get("sub_query_total_docs", 0) + \
                 doc_count_info.get("rewrite_query_doc_count", 0) + \
                 doc_count_info.get("multi_query_total_docs", 0) + \
                 doc_count_info.get("superordinate_query_doc_count", 0) + \
                 doc_count_info.get("hypothetical_answer_doc_count", 0)
    doc_count_info["total_docs"] = total_docs

    perf_info = {
        "stage": "_recall",
        "duration": elapsed_time,
        "timestamp": time.time(),
        "doc_count_info": doc_count_info
    }
    new_state["performance"].append(perf_info)

    return new_state


def _filter_enhance(agent_state: AgentState, agent_config: AgentConfig, embeddings_model: Embeddings,
                    llm: BaseChatModel) -> AgentState:
    # 1. 并行过滤
    def _parallel_filter(queries_list: List[List[Document]]) -> List[List[Document]]:
        new_multi_query_results = []

        if queries_list and len(queries_list) > 0:
            for query_results in queries_list:
                if len(query_results) > 0:
                    single_query = query_results[0].metadata["query"]
                    filtered = _filter_by_similar(query=single_query, docs=query_results,
                                                  agent_config=agent_config, embeddings_model=embeddings_model,
                                                  llm=llm)
                    if filtered is None:
                        filtered = []
                    elif not isinstance(filtered, list):
                        filtered = [filtered]

                    new_multi_query_results.append(filtered)

        return new_multi_query_results

    # 2. 判断是否拆分为了子问题，如果是只处理之问题检索
    new_sub_query_results = []
    sub_query_results = agent_state.get("sub_query_results", [])
    if sub_query_results and len(sub_query_results) > 0:
        new_sub_query_results = invoke_with_timing(
            func=_parallel_filter,
            queries_list=sub_query_results,
            stage_name="filter_sub_queries",
            state=agent_state
        )

    new_rewrite_query_docs = []
    rewrite_query_docs = agent_state.get("rewrite_query_docs", [])
    if rewrite_query_docs and len(rewrite_query_docs) > 0:
        query = rewrite_query_docs[0].metadata["query"]
        new_rewrite_query_docs = invoke_with_timing(
            func=_filter_by_similar,
            query=query,
            docs=rewrite_query_docs,
            agent_config=agent_config,
            embeddings_model=embeddings_model,
            llm=llm,
            stage_name="filter_rewrite_query",
            state=agent_state
        )

    new_multi_query_docs = []
    multi_query_docs = agent_state.get("multi_query_docs", [])
    if multi_query_docs and len(multi_query_docs) > 0:
        new_multi_query_docs = invoke_with_timing(
            func=_parallel_filter,
            queries_list=multi_query_docs,
            stage_name="filter_multi_queries",
            state=agent_state
        )

    new_superordinate_query_docs = []
    superordinate_query_docs = agent_state.get("superordinate_query_docs", [])
    if superordinate_query_docs and len(superordinate_query_docs) > 0:
        query = superordinate_query_docs[0].metadata["query"]
        new_superordinate_query_docs = invoke_with_timing(
            func=_filter_by_similar,
            query=query,
            docs=superordinate_query_docs,
            agent_config=agent_config,
            embeddings_model=embeddings_model,
            llm=llm,
            stage_name="filter_superordinate_query",
            state=agent_state
        )

    new_hypothetical_answer_docs = []
    hypothetical_answer_docs = agent_state.get("hypothetical_answer_docs", [])
    if hypothetical_answer_docs and len(hypothetical_answer_docs) > 0:
        query = hypothetical_answer_docs[0].metadata["query"]
        new_hypothetical_answer_docs = invoke_with_timing(
            func=_filter_by_similar,
            query=query,
            docs=hypothetical_answer_docs,
            agent_config=agent_config,
            embeddings_model=embeddings_model,
            llm=llm,
            stage_name="filter_hypothetical_answer",
            state=agent_state
        )

    agent_state["sub_query_results"] = new_sub_query_results
    agent_state["rewrite_query_docs"] = new_rewrite_query_docs
    agent_state["multi_query_docs"] = new_multi_query_docs
    agent_state["superordinate_query_docs"] = new_superordinate_query_docs
    agent_state["hypothetical_answer_docs"] = new_hypothetical_answer_docs

    return agent_state


def _filter_by_similar(query: str, docs: List[Document], agent_config: AgentConfig, embeddings_model: Embeddings,
                       llm: BaseChatModel) -> List[Document]:
    """
    压缩过滤-过滤低相关文档
    压缩过滤-过滤无关内容
    冗余过滤-基于向量引擎
    :param query:
    :param docs:
    :return:
    """
    low_correction_threshold = agent_config.low_correction_threshold
    redundant_threshold = agent_config.redundant_threshold
    embeddings_model = embeddings_model

    # 1.压缩过滤-过滤低相关文档
    if agent_config.filter_low_correction_doc_embeddings_enabled:
        docs = filter_low_correction_doc_embeddings(query, docs, embeddings_model, low_correction_threshold)
    elif agent_config.filter_low_correction_doc_llm_enabled:
        docs = filter_low_correction_doc_llm(query, docs, llm)

    # 2.压缩过滤-过滤无关内容
    if agent_config.filter_low_correction_content_enabled:
        docs = filter_low_correction_content(query, docs, llm)

    # 3.冗余过滤-基于向量引擎
    if agent_config.filter_redundant_doc_embeddings_enabled:
        docs = filter_redundant_doc_embeddings(query, docs, embeddings_model, redundant_threshold)

    return docs


def _sort_enhance(agent_state: AgentState, agent_config: AgentConfig, reranker: DashScopeRerank) -> AgentState:
    logger.info('\n' + '-' * 100)

    docs = []
    # 1.收集所有的文档
    if agent_config.generate_sub_queries_enabled:
        doc_list_list = agent_state.get("sub_query_results", [])
        for doc_list in doc_list_list:
            docs.extend(doc_list)
    if agent_config.query_rewrite_enabled:
        docs.extend(agent_state.get("rewrite_query_docs", []))
    if agent_config.generate_multi_queries_enabled:
        doc_list_list = agent_state.get("multi_query_docs", [])
        for doc_list in doc_list_list:
            docs.extend(doc_list)
    if agent_config.generate_superordinate_query_enabled:
        docs.extend(agent_state.get("superordinate_query_docs", []))
    if agent_config.generate_hypothetical_answer_enabled:
        docs.extend(agent_state.get("hypothetical_answer_docs", []))

    # 2.排序
    docs = invoke_with_timing(
        func=_sort_deduplicate_and_rank,
        docs=docs,
        agent_config=agent_config,
        reranker=reranker,
        stage_name="sort_deduplicate_and_rank",
        state=agent_state
    )
    agent_state["docs"] = docs

    return agent_state


def _sort_deduplicate_and_rank(docs: List[Document], agent_config: AgentConfig, reranker: DashScopeRerank,
                               agent_state: AgentState = None) -> List[
    Document]:
    # 1.基于模型重排序
    if agent_config.sort_docs_cross_encoder_enabled:
        docs = invoke_with_timing(
            func=sort_docs_cross_encoder,
            docs=docs,
            reranker=reranker,
            stage_name="sort_docs_cross_encoder",
            state=agent_state
        )

    # 2.过滤重复文档 如果做了冗余过滤，逻辑上是没有必要了
    docs_set = set()
    new_docs = []
    for doc in docs:
        if doc.metadata["id"] not in docs_set:
            new_docs.append(doc)
            docs_set.add(doc.metadata["id"])

    # 3.基于长文本的重排序
    if agent_config.sort_docs_by_loss_of_location_enabled:
        new_docs = invoke_with_timing(
            func=sort_docs_by_loss_of_location,
            docs=new_docs,
            stage_name="sort_docs_by_loss_of_location",
            state=agent_state
        )

    return new_docs


def _answer(state: AgentState, llm: BaseChatModel, show_debug) -> AgentState:
    answer_state = AnswerState(query=state["curr_input"], docs=state["docs"], )

    # 1. 生成回答
    answer_state = generate_answer(answer_state, llm, show_debug, agent_state=state)
    answer_str = answer_state.get("answer", "")
    state["final_answer"] = answer_str

    # 2. 存储主对话(多轮补充后的用户最后输入和最终答案)
    state["dialogue_messages"].append(
        HumanMessage(content=state["curr_input"])
    )
    state["dialogue_messages"].append(
        AIMessage(content=answer_str)
    )

    # 3. 存储问答字符串
    if len(state["asking_messages"][-1]) > 0:
        question = state["background_info"]  # 问题和问题补充后的总结，也是后续操作的关键
        question_answer = question + "\n" + answer_str
        state["multi_summary"].append(question_answer)
    return state


def route_ask_again(state: AgentState) -> str:
    """路由函数：判断是否需要继续向用户追问"""
    return (
        "ask" if state["ask_obj"].need_ask and state["curr_ask_num"] < state["max_ask_num"]
        else "pass"
    )


def _history_clean(state: AgentState, keep_count: int = 10):
    # dialogue_messages: List[BaseMessage]  # 主对话历史
    # asking_messages: List[List[BaseMessage]]  # 每一轮追问形成一组子对话
    # multi_summary: List[str]  # 多轮对话摘要列表
    # performance: List[Any]  # 调试 / 性能信息
    state["dialogue_messages"] = state["dialogue_messages"][-keep_count:]
    state["asking_messages"] = state["asking_messages"][-keep_count:]
    state["multi_summary"] = state["multi_summary"][-keep_count // 2:]
    # 性能信息不截取，保留完整的性能监控数据
    state["performance"] = []
