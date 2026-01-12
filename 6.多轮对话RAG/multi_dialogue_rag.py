"""多轮对话RAG实现"""
import logging
import re
import time
import traceback
from typing import List, Dict, Union

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from config.models import AppConfig, SearchRequest, SingleSearchRequest, FusionSpec
from knowledge_base import KnowledgeBase
from prompts import get_prompt_template
from rag_base import BasicRAG
from retriever import KnowledgeRetriever
from utils import create_llm_client, ESTIMATE_FUNCTION_REGISTRY

logger = logging.getLogger(__name__)


class MultiDialogueRag(BasicRAG):
    """
    多轮对话医疗RAG系统

    功能概述：
    -----------
    1. 多轮对话管理：通过session_id区分不同的会话，每个会话维护独立的对话历史
    2. 历史压缩：当对话历史过长时，自动将旧对话压缩成摘要，释放token空间
    3. 查询改写：基于对话历史，将当前用户问题改写为更完整的独立查询
    4. 混合检索：结合稀疏检索（BM25）和稠密检索（向量相似度）获取相关文档
    5. 动态上下文管理：根据token预算动态调整上下文内容
    """

    def __init__(self, config: AppConfig, search_config: SearchRequest = None):
        """初始化多轮对话RAG系统"""
        super().__init__(config, search_config)

        # 初始化核心组件
        self.knowledge_base = KnowledgeBase(config)
        self.retriever = KnowledgeRetriever(self.knowledge_base, self.search_config)
        self.llm = create_llm_client(config.llm)

        # ========== 会话存储 & 摘要存储 ==========
        # 存储各会话的对话历史消息对象
        # 类型: Dict[str, ChatMessageHistory]
        # 示例: {
        #     "session_001": ChatMessageHistory(messages=[
        #         HumanMessage(content="高血压有什么症状？"),
        #         AIMessage(content="高血压的常见症状包括...")
        #     ]),
        #     "session_002": ChatMessageHistory(messages=[...])
        # }
        self._histories: Dict[str, ChatMessageHistory] = {}

        # 存储各会话的历史消息字符长度与实际token数统计，用于计算token估算比率
        # 类型: Dict[str, Dict[str, List[int]]]
        # 示例: {
        #     "session_001": {
        #         "msg_len": [15, 50, 20, 80],           # 每条消息的字符长度
        #         "msg_token_len": [5, 18, 7, 28]        # 每条消息实际消耗的token数
        #     }
        # }
        self._token_meta_store = {}

        # 存储各会话的对话摘要（历史压缩后生成的摘要文本）
        # 每次触发历史压缩时：
        #   将新摘要 summary 与已有摘要 prev 合并
        #   更新为一条累积的摘要字符串
        # 类型: Dict[str, str]
        # 示例: {
        #     "session_001": "用户询问了高血压的症状和饮食建议，医生解释了高血压的常见表现...",
        #     "session_002": "用户咨询了糖尿病的治疗方案..."
        # }
        self._running_summaries: Dict[str, str] = {}

        # 存储各会话的摘要列表，用于控制缓存数量
        # 类型: Dict[str, List[str]]
        # 示例: {
        #     "session_001": ["第一次摘要...", "第二次摘要...", "第三次摘要..."],
        #     "session_002": ["第一次摘要..."]
        # }
        self._summary_cache_list: Dict[str, List[str]] = {}

        # 缓存时间戳记录，用于会话过期清理
        # 类型: Dict[str, float]
        # 示例: {
        #     "session_001": 1736612400.123,  # 最后访问时间戳（秒）
        #     "session_002": 1736612500.456
        # }
        self._cache_timestamps: Dict[str, float] = {}

        # 动态生成上下文的变量
        self._system_prompt_text_len = len(get_prompt_template("dialogue_rag")["system"])
        self._user_prompt_text_len = len(get_prompt_template("dialogue_rag")["user"])
        self._histories_prompt_text_len = 0
        self.avg_tokens_per_char = 1e-5

        # 设置提示词模板和处理链
        self.dialogue_rag_prompt = self._setup_dialogue_rag_prompt()
        self._setup_chain()

        logger.info("多轮对话 RAG 初始化完成")

    def _create_default_search_config(self) -> SearchRequest:
        """创建默认检索配置"""
        # 从配置中读取检索字段配置
        if self.config.rag.anns_fields:
            requests = [
                SingleSearchRequest(**f.model_dump())
                for f in self.config.rag.anns_fields
            ]
        else:
            # 默认配置
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
            "chunk", "parent_chunk", "summary", "questions",
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

    # ---------- 缓存管理 ----------

    def _clean_expired_cache(self):
        """清理过期的会话历史和摘要缓存"""
        cache_time_minutes = self.config.multi_dialogue_rag.cache_time
        if cache_time_minutes == 0:
            return  # cache_time=0 表示不超时

        current_time = time.time()
        expire_threshold = cache_time_minutes * 60  # 转换为秒

        expired_sessions = []
        for session_id, last_access_time in list(self._cache_timestamps.items()):
            if current_time - last_access_time > expire_threshold:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self._histories.pop(session_id, None)
            self._running_summaries.pop(session_id, None)
            self._summary_cache_list.pop(session_id, None)
            self._token_meta_store.pop(session_id, None)
            self._cache_timestamps.pop(session_id, None)
            if self.config.multi_dialogue_rag.console_debug:
                logger.info(f"[{session_id}] 会话缓存已过期，已清理")

    # ---------- 历史获取器 ----------

    def _get_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建指定会话的对话历史对象"""
        if session_id not in self._histories:
            self._histories[session_id] = ChatMessageHistory()
            self._running_summaries[session_id] = ""
            self._summary_cache_list[session_id] = []
        # 更新缓存时间戳
        self._cache_timestamps[session_id] = time.time()
        return self._histories[session_id]

    def _setup_dialogue_rag_prompt(self) -> ChatPromptTemplate:
        """构建多轮对话RAG的提示词模板"""
        base = get_prompt_template("dialogue_rag")

        system_msg = base["system"]
        user_msg = base["user"]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                MessagesPlaceholder(variable_name="history"),
                ("human", user_msg),
            ]
        )
        return prompt

    def _avg_estimate_over_max_token(self, session_id: str, exist_chars: int):
        """使用平均值方法估算当前上下文是否超过token预算"""
        meta = self._token_meta_store.get(session_id)
        if not meta:
            return False

        msg_len = meta["msg_len"]
        msg_token_len = meta["msg_token_len"]
        self.avg_tokens_per_char = sum(msg_token_len) / sum(msg_len)

        avg_char_len = sum(msg_len) / max(1, len(msg_len))
        predict_token = int(self.avg_tokens_per_char * avg_char_len)

        curr_all_token = int(predict_token + exist_chars * self.avg_tokens_per_char)
        if curr_all_token > self.config.multi_dialogue_rag.llm_max_token * self.config.multi_dialogue_rag.max_token_threshold:
            return True
        else:
            return False

    # ---------- 上下文压缩 ----------

    def _maybe_compress_history(self, session_id: str):
        """检查并可能压缩对话历史"""
        hist = self._histories.get(session_id)

        if not hist:
            return

        total_chars = "\n".join([m.content for m in hist.messages if hasattr(m, "content")])
        total_chars_len = len(total_chars)

        if self.config.multi_dialogue_rag.estimate_token_fun != "avg":
            try:
                estimate_token = ESTIMATE_FUNCTION_REGISTRY[self.config.multi_dialogue_rag.estimate_token_fun](
                    total_chars)
                if estimate_token < self.config.multi_dialogue_rag.llm_max_token * self.config.multi_dialogue_rag.max_token_threshold:
                    return
            except Exception as e:
                logger.error("注册的估计函数错误，回退到默认avg实现...")
                print(traceback(e))
                if not self._avg_estimate_over_max_token(session_id=session_id, exist_chars=total_chars_len):
                    return
        elif not self._avg_estimate_over_max_token(session_id=session_id, exist_chars=total_chars_len):
            return

        self._get_summary(session_id)

    def _get_summary(self, session_id: str):
        """将旧对话消息压缩成摘要"""
        hist = self._histories.get(session_id)

        if self.config.multi_dialogue_rag.console_debug:
            logger.warning(f"[{session_id}] 对话过长，需要生成摘要...")

        keep_count = max(2, len(hist.messages) // self.config.multi_dialogue_rag.cut_dialogue_scale)
        n = len(hist.messages) - keep_count
        old_msgs = hist.messages[:n]

        # 构建摘要提示词，包含长度限制
        summary_prompt_template = get_prompt_template("summary")
        summary_system = summary_prompt_template["system"]
        summary_user = summary_prompt_template["user"]

        # 在用户提示词中添加长度限制
        max_length = self.config.multi_dialogue_rag.summary_max_length
        summary_user_with_limit = f"{summary_user}\n\n请确保摘要内容简洁，控制在{max_length}字符以内。"

        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", summary_system),
            MessagesPlaceholder("history"),
            ("human", summary_user_with_limit)
        ])
        summary_result: AIMessage = (summarize_prompt | self.llm).invoke({"history": old_msgs})

        # 截断摘要到指定长度
        summary = re.sub(r"<\|.*?\|>\s*", "", summary_result.content, flags=re.DOTALL).strip()
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        dur = summary_result.response_metadata.get("total_duration", 0) / 1e9
        tokens = summary_result.usage_metadata["total_tokens"]

        if self.config.multi_dialogue_rag.console_debug:
            logger.warning(f"[{session_id}] 摘要生成完毕，耗时：{dur} s，使用tokens：{tokens}\n摘要文本：\n{summary}")

        # 管理摘要缓存列表
        if session_id not in self._summary_cache_list:
            self._summary_cache_list[session_id] = []

        # 添加新摘要到缓存列表
        self._summary_cache_list[session_id].append(summary)

        # 检查是否超过缓存数量限制
        max_cache_count = self.config.multi_dialogue_rag.summary_max_cache_count
        if len(self._summary_cache_list[session_id]) > max_cache_count:
            # 只保留最近的N次摘要
            if self.config.multi_dialogue_rag.console_debug:
                logger.info(
                    f"[{session_id}] 摘要缓存数量({len(self._summary_cache_list[session_id])})超过限制({max_cache_count})，保留最近的{max_cache_count}次摘要")
            self._summary_cache_list[session_id] = self._summary_cache_list[session_id][-max_cache_count:]

        # 合并所有缓存的摘要
        merged_summary = "\n".join(self._summary_cache_list[session_id])
        self._running_summaries[session_id] = merged_summary

        hist.messages = hist.messages[n:]

    @staticmethod
    def _strip_think_get_tokens(msg: AIMessage):
        """
        处理LLM返回消息，移除思考标签并提取token统计信息

        Args:
            msg: LLM返回的AI消息对象

        Returns:
            包含处理后的消息内容、消息长度、token数、生成时间的字典

        Notes:
            - usage_metadata["output_tokens"]: LangChain自动从LLM响应中提取的输出token数
            - response_metadata["total_duration"]: LangChain自动从LLM响应中提取的总耗时（纳秒）
        """
        text = msg.content
        msg_len = len(msg.content)
        msg_token_len = msg.usage_metadata["output_tokens"]
        dur = msg.response_metadata.get("total_duration", 0) / 1e9
        return {
            "msg": re.sub(r"<\|.*?\|>\s*", "", text, flags=re.DOTALL).strip(),
            "msg_len": msg_len,
            "msg_token_len": msg_token_len,
            "generate_time": dur
        }

    def _build_document_context(
            self,
            documents: List[Document],
            rewritten_query: str,
            session_id: str,
            history_msgs: List[Union[AIMessage, HumanMessage]]
    ) -> str:
        """根据token预算构建文档上下文字符串"""
        remain_token = self.config.multi_dialogue_rag.llm_max_token
        his_text = "\n".join(
            getattr(m, "content", "") for m in history_msgs if hasattr(m, "content")
        )

        user_text = get_prompt_template("dialogue_rag")["user"].format(
            llm_rewritten_content=rewritten_query,
            all_document_str=""
        )
        summaries_text = self._running_summaries[session_id]
        system_text = get_prompt_template("dialogue_rag")["system"].format(running_summary=summaries_text)
        all_chars = his_text + system_text + user_text

        parts = []
        used = 0

        if self.config.multi_dialogue_rag.estimate_token_fun != "avg":
            all_token = ESTIMATE_FUNCTION_REGISTRY[self.config.multi_dialogue_rag.estimate_token_fun](all_chars)
            remain_token -= all_token + self.config.multi_dialogue_rag.llm_max_token * 0.01
        else:
            all_prompt_chars_len = len(all_chars)
            remain_token -= all_prompt_chars_len + self.config.multi_dialogue_rag.llm_max_token * 0.01

        for idx, d in enumerate(documents):
            header = f"## 文档{idx + 1}：\n"
            body = d.page_content or ""

            if self.config.multi_dialogue_rag.estimate_token_fun != "avg":
                header_tokens = ESTIMATE_FUNCTION_REGISTRY[self.config.multi_dialogue_rag.estimate_token_fun](header)
                body_tokens = ESTIMATE_FUNCTION_REGISTRY[self.config.multi_dialogue_rag.estimate_token_fun](body)
            else:
                header_tokens = self.avg_tokens_per_char * len(header)
                body_tokens = self.avg_tokens_per_char * len(body)

            if used + header_tokens + body_tokens <= remain_token:
                parts.append(header + body + "\n")
                used += header_tokens + body_tokens
            else:
                if self.config.multi_dialogue_rag.console_debug:
                    logger.warning(
                        f"根据给定的token估计方法，预估无法完成全部文档编码，文档{idx + 1}被截断，后续文档将无法被放入上下文...")

                remain = remain_token - used - header_tokens
                if remain > 0:
                    keep_chars = max(0, int(remain / max(0.1, self.avg_tokens_per_char)))
                    if keep_chars > 0:
                        parts.append(header + body[:keep_chars] + "\n...[内容已截断]\n")
                        used = remain_token
                break

        return "".join(parts)

    # ---------- 构建多轮 RAG 链 ----------

    def _setup_chain(self):
        """构建完整的多轮对话RAG处理链"""

        # 查询改写链
        rewrite_template = ChatPromptTemplate.from_messages([
            ("system", get_prompt_template("rewriter")["system"]),
            MessagesPlaceholder("history"),
            ("human", get_prompt_template("rewriter")["user"])
        ])
        rewritten_query_chain = rewrite_template | self.llm | RunnableLambda(self._strip_think_get_tokens)

        def do_retrieve(inputs: dict):
            """使用改写后的问题进行检索"""
            logger.info(f"改写后的问题: {inputs['llm_rewritten_query']['msg']}")
            return self.retriever.invoke({"input": inputs["llm_rewritten_query"]["msg"]})

        def do_format(inputs: dict) -> str:
            """格式化文档上下文"""
            documents: List[Document] = inputs["milvus_result"]["documents"]
            all_document_str = self._build_document_context(
                documents=documents,
                rewritten_query=inputs["llm_rewritten_query"]["msg"],
                session_id=inputs.get("session_id", "default"),
                history_msgs=inputs.get("history", [])
            )
            return {**inputs, "all_document_str": all_document_str,
                    "llm_rewritten_content": inputs["llm_rewritten_query"]["msg"]}

        # 答案生成链
        out_answer = (
                RunnableLambda(do_format)
                | self.dialogue_rag_prompt
                | self.llm
                | RunnableLambda(self._strip_think_get_tokens)
        )

        # ========== 核心处理链 ==========
        # core_chain 构建了完整的多轮对话RAG处理流水线
        # 数据流向: 用户输入 -> 查询改写 -> 文档检索 -> 答案生成 -> 输出提取

        # ---------- 第一步: 查询改写 ----------
        # 功能: 基于对话历史，将用户原始问题改写为更完整、上下文清晰的独立查询
        # 输入格式: {"original_input": str, "running_summary": str, "session_id": str, "history": List[Message]}
        # 输出格式: {
        #     "original_input": str,           # 原始用户问题
        #     "running_summary": str,          # 历史对话摘要
        #     "session_id": str,               # 会话ID
        #     "history": List[Message],        # 对话历史消息列表
        #     "llm_rewritten_query": {        # 新增字段：改写后的查询结果
        #         "msg": str,                   # 改写后的查询文本
        #         "msg_len": int,               # 查询文本字符长度
        #         "msg_token_len": int,        # 改写消耗的token数
        #         "generate_time": float        # 改写耗时(秒)
        #     }
        # }
        core_chain = (
                RunnablePassthrough.assign(llm_rewritten_query=rewritten_query_chain)
                .with_config(run_name="rewritten_query")

                # ---------- 第二步: 文档检索 ----------
                # 功能: 使用改写后的问题从知识库中检索相关文档
                # 输入格式: {
                #     "original_input": str,
                #     "running_summary": str,
                #     "session_id": str,
                #     "history": List[Message],
                #     "llm_rewritten_query": {...}
                # }
                # 输出格式: {
                #     "original_input": str,
                #     "running_summary": str,
                #     "session_id": str,
                #     "history": List[Message],
                #     "llm_rewritten_query": {...},
                #     "milvus_result": {                 # 新增字段：检索结果
                #         "documents": List[Document],   # 检索到的文档列表
                #         "search_time": float           # 检索耗时(秒)
                #     }
                # }
                | RunnablePassthrough.assign(milvus_result=RunnableLambda(do_retrieve))
                .with_config(run_name="search_documents")

                # ---------- 第三步: 答案生成 ----------
                # 功能: 基于改写查询、检索文档和对话历史，生成最终答案
                # 输入格式: {
                #     "original_input": str,
                #     "running_summary": str,
                #     "session_id": str,
                #     "history": List[Message],
                #     "llm_rewritten_query": {...},
                #     "milvus_result": {...}
                # }
                # 输出格式: {
                #     "original_input": str,
                #     "running_summary": str,
                #     "session_id": str,
                #     "history": List[Message],
                #     "llm_rewritten_query": {...},
                #     "milvus_result": {...},
                #     "llm_out_result": {                # 新增字段：LLM生成的结果
                #         "msg": str,                     # 生成的答案文本
                #         "msg_len": int,                 # 答案字符长度
                #         "msg_token_len": int,          # 生成消耗的token数
                #         "generate_time": float          # 生成耗时(秒)
                #     }
                # }
                | RunnablePassthrough.assign(llm_out_result=out_answer)
                .with_config(run_name="generate")

                # ---------- 第四步: 提取最终答案 ----------
                # 功能: 将llm_out_result中的答案提取到顶层，方便最终输出
                # 输入格式: {
                #     "original_input": str,
                #     "running_summary": str,
                #     "session_id": str,
                #     "history": List[Message],
                #     "llm_rewritten_query": {...},
                #     "milvus_result": {...},
                #     "llm_out_result": {...}
                # }
                # 输出格式: {
                #     "original_input": str,
                #     "running_summary": str,
                #     "session_id": str,
                #     "history": List[Message],
                #     "llm_rewritten_query": {...},
                #     "milvus_result": {...},
                #     "llm_out_result": {...},
                #     "answer": str                       # 新增字段：最终答案(从llm_out_result.msg提取)
                # }
                | RunnableLambda(lambda x: {**x, "answer": x["llm_out_result"]["msg"]})
        )

        def _get_history_wrapper(session_id: str):
            """包装历史获取器"""
            return self._get_history(session_id)

        # 支持历史管理的链
        self.rag_chain = RunnableWithMessageHistory(
            core_chain,
            _get_history_wrapper,
            input_messages_key="original_input",
            history_messages_key="history",
            output_messages_key="answer",
        ).with_config(run_name="rag")

    def _update_tokens_metadata(
            self,
            answer_result: dict,
            session_id: str
    ):
        """更新token元数据，用于更准确的token估算"""
        if self._token_meta_store.get(session_id) is not None:
            self._token_meta_store[session_id]["msg_len"].append(answer_result["llm_rewritten_query"]["msg_len"])
            self._token_meta_store[session_id]["msg_token_len"].append(
                answer_result["llm_rewritten_query"]["msg_token_len"])
            self._token_meta_store[session_id]["msg_len"].append(answer_result["llm_out_result"]["msg_len"])
            self._token_meta_store[session_id]["msg_token_len"].append(answer_result["llm_out_result"]["msg_token_len"])
        else:
            msg_len = [
                answer_result["llm_rewritten_query"]["msg_len"],
                answer_result["llm_out_result"]["msg_len"]
            ]
            msg_token_len = [
                answer_result["llm_rewritten_query"]["msg_token_len"],
                answer_result["llm_out_result"]["msg_token_len"]
            ]
            self._token_meta_store[session_id] = {
                "msg_len": msg_len,
                "msg_token_len": msg_token_len
            }

    def answer(
            self,
            query: str,
            return_document: bool = False,
            session_id: str = "default"
    ) -> Union[str, Dict[str, Union[str, List[Document]]]]:
        """
        处理用户问题并返回答案（对外主API）

        Args:
            query: 用户问题
            return_document: 是否返回检索到的文档
            session_id: 会话ID

        Returns:
            答案字典
        """
        logger.info(f"[{session_id}] 问题: {query}")
        try:
            # 0) 清理过期缓存
            self._clean_expired_cache()

            # 1) 可能压缩旧历史
            self._maybe_compress_history(session_id)

            # 2) 运行RAG处理链
            result = self.rag_chain.invoke(
                {
                    "original_input": query,
                    "running_summary": self._running_summaries.get(session_id, ""),
                    "session_id": session_id
                },
                config={"configurable": {"session_id": session_id}}
            )

            # 3) 更新token计数
            self._update_tokens_metadata(answer_result=result, session_id=session_id)

            answer = result.get("answer", "抱歉，根据提供的资料无法回答您的问题。")

            if return_document:
                return {
                    "answer": answer,
                    "documents": result["milvus_result"]["documents"],
                    "search_time": result["milvus_result"]["search_time"],
                    "rewriten_generate_time": result["llm_rewritten_query"]["generate_time"],
                    "out_generate_time": result["llm_out_result"]["generate_time"]
                }
            return {
                "answer": answer,
                "search_time": result["milvus_result"]["search_time"],
                "rewriten_generate_time": result["llm_rewritten_query"]["generate_time"],
                "out_generate_time": result["llm_out_result"]["generate_time"]
            }

        except Exception as e:
            logger.exception(f"[{session_id}] RAG处理失败: {e}")
            error_msg = "抱歉，处理您的问题时出现错误，请稍后再试。"

            if return_document:
                return {
                    "answer": error_msg,
                    "documents": [],
                    "search_time": -1,
                    "rewriten_generate_time": -1,
                    "out_generate_time": -1
                }
            return {
                "answer": error_msg,
                "search_time": -1,
                "rewriten_generate_time": -1,
                "out_generate_time": -1
            }

    def update_search_config(self, search_config: SearchRequest):
        """更新检索配置并重建RAG链"""
        self.search_config = search_config
        self.retriever = KnowledgeRetriever(self.knowledge_base, search_config)
        self._setup_chain()
        logger.info(f"搜索配置已更新: {search_config}")
