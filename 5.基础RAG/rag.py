"""基础RAG实现"""
import logging
import re
import traceback
from typing import List, Dict, Any, Union, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage

from config.models import AppConfig, SearchRequest, SingleSearchRequest, FusionSpec
from config.loader import ConfigLoader
from knowledge_base import MedicalKnowledgeBase
from retriever import MedicalRetriever
from utils import create_llm_client, format_documents
from prompts import get_prompt_template


logger = logging.getLogger(__name__)


class SimpleRAG:
    """基础医疗RAG系统"""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        config_path: Optional[str] = None,
        search_config: Optional[SearchRequest] = None
    ):
        """
        初始化RAG系统

        Args:
            config: 应用配置
            config_path: 配置文件路径
            search_config: 检索配置
        """
        # 加载配置
        if config is None:
            if config_path is None:
                config_loader = ConfigLoader()
            else:
                config_loader = ConfigLoader(config_path)
            self.config = config_loader.config
        else:
            self.config = config

        # 初始化检索配置
        if search_config is None:
            self.search_config = self._create_default_search_config()
        else:
            self.search_config = search_config

        # 初始化知识库和检索器
        self.knowledge_base = MedicalKnowledgeBase(self.config)
        self.retriever = MedicalRetriever(self.knowledge_base, self.search_config)

        # 初始化LLM
        self.llm = create_llm_client(self.config.llm)

        # 设置prompt模板
        self.prompt = self._setup_prompt()

        # 构建RAG链
        self._setup_chain()

        logger.info("RAG系统初始化完成")

    def _create_default_search_config(self) -> SearchRequest:
        """创建默认检索配置"""
        # 从配置中读取默认字段配置
        if self.config.rag.default_fields:
            requests = [
                SingleSearchRequest(**f.model_dump())
                for f in self.config.rag.default_fields
            ]
        else:
            # 默认配置（根据simple_rag.yaml中的默认字段）
            requests = [
                SingleSearchRequest(
                    anns_field="chunk_dense",
                    metric_type="COSINE",
                    search_params={"ef": 64},
                    limit=50,
                    expr=""
                ),
                SingleSearchRequest(
                    anns_field="parent_chunk_dense",
                    metric_type="COSINE",
                    search_params={"ef": 64},
                    limit=50,
                    expr=""
                ),
                SingleSearchRequest(
                    anns_field="questions_dense",
                    metric_type="COSINE",
                    search_params={"ef": 64},
                    limit=50,
                    expr=""
                ),
                SingleSearchRequest(
                    anns_field="chunk_sparse",
                    metric_type="IP",
                    search_params={"drop_ratio_search": 0.0},
                    limit=50,
                    expr=""
                )
            ]

        fusion = FusionSpec(
            method=self.config.rag.fusion.method,
            k=self.config.rag.fusion.k,
            weights=list(self.config.rag.fusion.weights.values())
        )

        output_fields = self.config.rag.output_fields or [
            "chunk", "parent_chunk", "summary", "questions", "document",
            "source", "source_name", "lt_doc_id", "chunk_id", "hash_id"
        ]

        return SearchRequest(
            query="",
            collection_name=self.config.milvus.collection_name,
            requests=requests,
            output_fields=output_fields,
            fuse=fusion,
            top_k=self.config.rag.top_k,
            limit=self.config.rag.limit or 5
        )

    def _setup_prompt(self) -> ChatPromptTemplate:
        """设置提示模板"""
        template = get_prompt_template("basic_rag")

        if isinstance(template, dict):
            prompt = ChatPromptTemplate.from_messages([
                ("system", template["system"]),
                ("human", template["user"]),
            ])
        else:
            prompt = ChatPromptTemplate.from_template(template)

        return prompt

    def _strip_think_and_time(self, msg: AIMessage) -> Dict[str, Any]:
        """
        清理LLM输出并统计时间

        Args:
            msg: LLM响应消息

        Returns:
            清理后的答案和生成时间
        """
        text = msg.content

        # 移除思考过程
        cleaned = re.sub(
            r"<think>.*?</think>\s*",
            "",
            text,
            flags=re.DOTALL
        )

        # 获取生成耗时（纳秒转秒）
        duration = msg.response_metadata.get("total_duration", 0) / 1e9

        return {
            "answer": cleaned.strip(),
            "generate_time": duration
        }

    def _setup_chain(self):
        """构建RAG链"""
        # 文档格式化
        def format_doc_str(inputs: dict) -> str:
            documents = inputs["retrieval_result"]["documents"]
            return format_documents(documents)

        # 检索阶段
        retrieve = RunnablePassthrough.assign(
            retrieval_result=self.retriever
        ).with_config(run_name="retrieve")

        # 文档格式化阶段
        format_docs = RunnablePassthrough.assign(
            all_document_str=RunnableLambda(format_doc_str)
        ).with_config(run_name="format_docs")

        # 生成阶段
        generate = (
            self.prompt.with_config(run_name="prompt")
            | self.llm.with_config(run_name="llm")
            | RunnableLambda(self._strip_think_and_time)
        )

        # 完整RAG链
        self.rag_chain = (
            retrieve
            | format_docs
            | RunnablePassthrough.assign(llm=generate)
        ).with_config(run_name="rag")

        logger.info("RAG链构建完成")

    def answer(
        self,
        query: str,
        return_document: bool = False
    ) -> Dict[str, Union[str, List[Document], float]]:
        """
        回答用户问题

        Args:
            query: 用户问题
            return_document: 是否返回检索到的文档

        Returns:
            包含答案和元数据的字典
        """
        logger.info(f"处理问题: {query}")

        try:
            result = self.rag_chain.invoke({"input": query})
            answer = result["llm"]["answer"]
            search_time = result["retrieval_result"]["search_time"]
            generate_time = result["llm"]["generate_time"]

            if return_document:
                return {
                    "answer": answer,
                    "documents": result["retrieval_result"]["documents"],
                    "search_time": search_time,
                    "generation_time": generate_time
                }

            return {
                "answer": answer,
                "search_time": search_time,
                "generation_time": generate_time
            }

        except Exception as e:
            logger.error(f"RAG处理失败: {e}")
            print(traceback.format_exc())

            error_msg = "抱歉，处理您的问题时出现错误，请稍后再试。"

            if return_document:
                return {
                    "answer": error_msg,
                    "documents": []
                }
            return {"answer": error_msg}

    def batch_answer(
        self,
        queries: List[str],
        return_document: bool = False
    ) -> List[Dict[str, Union[str, List[Document], float]]]:
        """
        批量回答问题

        Args:
            queries: 问题列表
            return_document: 是否返回检索到的文档

        Returns:
            回答列表
        """
        return [self.answer(q, return_document) for q in queries]

    def update_search_config(self, search_config: SearchRequest):
        """
        更新检索配置

        Args:
            search_config: 新的检索配置
        """
        self.search_config = search_config
        self.retriever = MedicalRetriever(self.knowledge_base, search_config)
        self._setup_chain()
        logger.info(f"搜索配置已更新")
