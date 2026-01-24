from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app_config import APPConfig
from enhance.agent_state import AgentState, RecallDocument
from enhance.filter_enhance import filter_low_correction_doc_embeddings, filter_low_correction_doc_llm, \
    filter_low_correction_content, filter_redundant_doc_embeddings
from enhance.recall_enhance import generate_multi_queries, generate_superordinate_query, generate_hypothetical_answer, \
    generate_sub_queries
from sentence_transformers import CrossEncoder

from enhance.sort_enhance import sort_docs_cross_encoder, sort_docs_by_loss_of_location


class DialogueAgent:
    def __init__(self, app_config: APPConfig, embeddings_model: Embeddings, llm: BaseChatModel,
                 reranker: CrossEncoder) -> None:
        self.app_config = app_config
        self.agent_config = app_config.agent
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.reranker = reranker

        # 代理总图
        self.agent_graph = None

    def answer(self, user_input: str) -> str:
        """回答用户输入"""
        return self.llm.as_retriever()(user_input)

    def _build_graph(self):

    def _recall_enhance(self, agent_state: AgentState):
        # 1.生成多个问题
        if self.agent_config.generate_multi_queries_enabled:
            agent_state = generate_multi_queries(agent_state, self.llm)

        # 2.生成上位词生成上位词    if self.agent_config.generate_multi_queries_enabled:
        if self.agent_config.generate_superordinate_query_enabled:
            agent_state = generate_superordinate_query(agent_state, self.llm)

        # 3.生成假设回答
        if self.agent_config.generate_hypothetical_answer_enabled:
            agent_state = generate_hypothetical_answer(agent_state, self.llm)

        # 4.拆分成子问题
        if self.agent_config.generate_sub_queries_enabled:
            agent_state = generate_sub_queries(agent_state, self.llm)

        return agent_state

    def _filter_enhance(self, agent_state: AgentState):
        new_sub_query_results = []
        sub_query_results = agent_state.get("sub_query_results", [])
        if sub_query_results:
            for sub_query_result in sub_query_results:
                if len(sub_query_result) > 0:
                    query = sub_query_result[0].metadata["query"]
                    sub_query_result = self._filter_enhance_by_query(query, sub_query_result)
                    new_sub_query_results.append(sub_query_result)

        new_rewrite_query_docs = []
        rewrite_query_docs = agent_state.get("rewrite_query_docs", [])
        if rewrite_query_docs and len(rewrite_query_docs) > 0:
            query = rewrite_query_docs[0].metadata["query"]
            new_rewrite_query_docs = self._filter_enhance_by_query(query, rewrite_query_docs)

        new_multi_query_docs = []
        multi_query_docs = agent_state.get("multi_query_docs", [])
        if multi_query_docs and len(multi_query_docs) > 0:
            for multi_query_doc in multi_query_docs:
                if len(multi_query_doc) > 0:
                    query = multi_query_doc[0].metadata["query"]
                    self._filter_enhance_by_query(query, multi_query_doc)
                    new_multi_query_docs.append(multi_query_doc)

        new_superordinate_query_docs = []
        superordinate_query_docs = agent_state.get("superordinate_query_docs", [])
        if superordinate_query_docs and len(superordinate_query_docs) > 0:
            query = superordinate_query_docs[0].metadata["query"]
            new_superordinate_query_docs = self._filter_enhance_by_query(query, superordinate_query_docs)

        new_hypothetical_answer_docs = []
        hypothetical_answer_docs = agent_state.get("hypothetical_answer_docs", [])
        if hypothetical_answer_docs and len(hypothetical_answer_docs) > 0:
            query = hypothetical_answer_docs[0].metadata["query"]
            new_hypothetical_answer_docs = self._filter_enhance_by_query(query, hypothetical_answer_docs)

        return {
            "sub_query_results": new_sub_query_results,
            "rewrite_query_docs": new_rewrite_query_docs,
            "multi_query_docs": new_multi_query_docs,
            "superordinate_query_docs": new_superordinate_query_docs,
            "hypothetical_answer_docs": new_hypothetical_answer_docs,
        }

    def _filter_enhance_by_query(self, query: str, docs: List[Document]) -> List[Document]:
        """
        压缩过滤-过滤低相关文档
        压缩过滤-过滤无关内容
        冗余过滤-基于向量引擎
        :param query:
        :param docs:
        :return:
        """
        low_correction_threshold = self.agent_config.low_correction_threshold
        redundant_threshold = self.agent_config.redundant_threshold
        embeddings_model = self.embeddings_model
        llm = self.llm

        # 1.压缩过滤-过滤低相关文档
        if self.agent_config.filter_low_correction_doc_embeddings_enabled:
            docs = filter_low_correction_doc_embeddings(query, docs, embeddings_model, low_correction_threshold)
        elif self.agent_config.filter_low_correction_doc_llm_enabled:
            docs = filter_low_correction_doc_llm(query, docs, llm)

        # 2.压缩过滤-过滤无关内容
        if self.agent_config.filter_low_correction_content_enabled:
            docs = filter_low_correction_content(query, docs, llm)

        # 3.冗余过滤-基于向量引擎
        if self.agent_config.filter_redundant_doc_embeddings_enabled:
            docs = filter_redundant_doc_embeddings(query, docs, embeddings_model, redundant_threshold)

        return docs

    def _sort_enhance(self, docs: List[Document]) -> List[Document]:
        if self.agent_config.sort_docs_cross_encoder_enabled:
            docs = sort_docs_cross_encoder(docs, self.reranker)

        if self.agent_config.sort_docs_by_loss_of_location_enabled:
            docs = sort_docs_by_loss_of_location(docs)

        return docs
