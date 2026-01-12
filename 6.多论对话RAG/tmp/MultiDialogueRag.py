from __future__ import annotations
import logging, re, traceback
from typing import List, Dict, Any, Optional, Union
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableMap
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ..config.models import *
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.HybridRetriever import MedicalHybridRetriever
from ..core.utils import create_llm_client
from ..prompts.templates import get_prompt_template
from .RagBase import BasicRAG
from .utils import ESTIMATE_FUNCTION_REGISTRY
import traceback
import os

logger = logging.getLogger(__name__)


class MultiDialogueRag(BasicRAG):
    """
    多轮对话医疗RAG系统

    功能概述：
    -----------
    该类实现了一个支持多轮对话的医疗领域RAG（检索增强生成）系统，主要功能包括：

    1. 多轮对话管理：通过session_id区分不同的会话，每个会话维护独立的对话历史
    2. 历史压缩：当对话历史过长时，自动将旧对话压缩成摘要，释放token空间
    3. 查询改写：基于对话历史，将当前用户问题改写为更完整的独立查询
    4. 混合检索：结合稀疏检索（BM25）和稠密检索（向量相似度）获取相关文档
    5. 动态上下文管理：根据token预算动态调整上下文内容，包括历史、摘要和文档
    6. Token估算：支持多种token估算方法，用于预测是否需要压缩历史

    核心流程：
    -----------
    用户问题 -> 查询改写 -> 混合检索 -> 文档上下文构建 -> LLM生成答案 -> 更新历史和token统计

    使用示例：
    -----------
    >>> config = AppConfig(...)
    >>> rag = MultiDialogueRag(config)
    >>> result = rag.answer("感冒吃什么药？", session_id="user_001")
    >>> print(result["answer"])
    """

    def __init__(self, config: AppConfig, search_config: SearchRequest = None):
        """
        初始化多轮对话RAG系统

        输入参数：
        ----------
        config : AppConfig 应用配置对象，包含LLM配置、数据库配置、多轮对话配置等
        search_config : SearchRequest, 可选
            检索配置对象，指定检索策略、返回数量等参数

        输出：无（初始化实例）

        功能说明：
        -----------
        1. 初始化知识库和混合检索器
        2. 创建LLM客户端
        3. 初始化会话历史存储、token统计存储、运行摘要存储
        4. 设置对话RAG提示词模板
        5. 构建完整的RAG处理链

        示例：
        -----------
        >>> from config.models import AppConfig, SearchRequest
        >>> config = AppConfig.load("config.yaml")
        >>> search_config = SearchRequest(top_k=5, hybrid_alpha=0.5)
        >>> rag = MultiDialogueRag(config, search_config)
        """
        super().__init__(config, search_config)

        # 配置LangSmith调试追踪（可选）
        if config.multi_dialogue_rag.smith_debug:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "rag-dev"

        # 初始化核心组件
        self.knowledge_base = MedicalHybridKnowledgeBase(config)  # 混合知识库（向量+BM25）
        self.self_retriever: BaseRetriever = MedicalHybridRetriever(self.knowledge_base, self.search_config)  # 混合检索器
        self.llm = create_llm_client(config.llm)  # LLM客户端

        # ---- 会话存储 & 摘要存储 ----
        self._histories: Dict[str, ChatMessageHistory] = {}  # session_id -> 对话历史对象
        self._token_meta_store = {}  # {session_id: {"msg_len": List[int], "msg_token_len": List[int]}}
        self._running_summaries: Dict[str, str] = {}  # session_id -> 累积的对话摘要

        # 用于动态生成上下文的变量
        self._system_prompt_text_len = len(get_prompt_template("dialogue_rag")["system"])
        self._user_prompt_text_len = len(get_prompt_template("dialogue_rag")["user"])
        self._histories_prompt_text_len = 0
        self.avg_tokens_per_char = 1e-5  # 平均token/字符比率，初始值设得很小以便首次交互

        # 设置提示词模板和处理链
        self.dialogue_rag_prompt = self._setup_dialogue_rag_prompt()
        self._setup_chain()

        logger.info("多轮对话 RAG 初始化完成")

    # ---------- 历史获取器 ----------

    def _get_history(self, session_id: str) -> ChatMessageHistory:
        """
        获取或创建指定会话的对话历史对象

        输入参数：
        ----------
        session_id : str
            会话ID，用于区分不同的用户会话
            输入示例："user_001", "session_abc123"

        输出：
        -------
        ChatMessageHistory
            LangChain的对话历史对象，存储该会话的所有消息

        功能说明：
        -----------
        1. 如果session_id已存在，返回对应的对话历史对象
        2. 如果session_id不存在，创建新的对话历史对象和空的摘要
        3. 对话历史对象会被RunnableWithMessageHistory使用，自动管理消息添加

        示例：
        -----------
        >>> hist = rag._get_history("user_001")
        >>> hist.add_user_message("你好")
        >>> hist.add_ai_message("你好！有什么可以帮助你的？")
        """
        if session_id not in self._histories:
            # 新会话：初始化对话历史和运行摘要
            self._histories[session_id] = ChatMessageHistory()
            self._running_summaries[session_id] = ""
        return self._histories[session_id]

    def _setup_dialogue_rag_prompt(self) -> ChatPromptTemplate:
        """
        构建多轮对话RAG的提示词模板

        输入：无

        输出：
        -------
        ChatPromptTemplate
            LangChain的提示词模板对象，包含system、history、human三个部分

        功能说明：
        -----------
        构建的消息结构如下：
        1. system: 系统提示词，包含角色定义、任务说明，以及可选的running_summary（长期摘要）
        2. history: 消息占位符，会被填充为最近的若干轮对话历史（短期记忆）
        3. human: 用户问题模板，包含改写后的问题和检索到的文档内容

        提示词模板示例：
        -----------
        System: 你是一个医疗助手...
        History: [之前的对话消息]
        Human: 参考以下资料回答问题：{llm_rewritten_content}，资料：{all_document_str}

        示例：
        -----------
        >>> prompt = rag._setup_dialogue_rag_prompt()
        >>> formatted = prompt.format_messages(
        ...     running_summary="用户咨询感冒症状...",
        ...     history=[HumanMessage("我感冒了"), AIMessage("请注意休息...")],
        ...     llm_rewritten_content="感冒后应该注意什么？",
        ...     all_document_str="文档1内容..."
        ... )
        """
        base = get_prompt_template("dialogue_rag")

        # 我们包装成统一消息结构：system(含长期摘要) + history(短期记忆) + human(当前问题)
        system_msg = base["system"]
        user_msg = base["user"]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),  # 系统提示，可能包含累积的对话摘要
                # 短期记忆（最近若干轮原文消息，会在运行时动态填充）
                MessagesPlaceholder(variable_name="history"),
                # 当前用户意图（包含改写后的问题和检索到的文档）
                ("human", user_msg),
            ]
        )
        return prompt

    def _avg_estimate_over_max_token(self, session_id: str, exist_chars: int):
        """
        使用平均值方法估算当前上下文是否超过token预算

        输入参数：
        ----------
        session_id : str
            会话ID，用于获取该会话的历史token统计
            输入示例："user_001"
        exist_chars : int
            当前已有上下文的字符数（不包含即将生成的回答）
            输入示例：5000

        输出：
        -------
        bool
            True: 需要压缩历史（超出预算）
            False: 不需要压缩（在预算内）

        功能说明：
        -----------
        基于历史交互数据，估算当前token使用情况：
        1. 计算历史交互的平均token/字符比率
        2. 预测本次回答可能生成的token数（基于历史平均长度）
        3. 判断总token数是否超过阈值的max_token_threshold倍
        4. 如果超过，返回True触发历史压缩

        算法步骤：
        -----------
        avg_tokens_per_char = sum(所有历史token) / sum(所有历史字符)
        predict_token = avg_tokens_per_char * 历史平均回答字符数
        curr_all_token = predict_token + exist_chars * avg_tokens_per_char
        return curr_all_token > llm_max_token * max_token_threshold

        示例：
        -----------
        >>> # 历史数据：5次交互，总共10000字符，2000 tokens
        >>> # 当前上下文8000字符，预计回答100字符
        >>> result = rag._avg_estimate_over_max_token("user_001", 8000)
        >>> # avg_tokens_per_char = 0.2, predict_token = 20
        >>> # curr_all_token = 20 + 8000*0.2 = 1620
        >>> # 如果阈值是4000*0.8=3200，则返回False
        """
        meta = self._token_meta_store.get(session_id)
        if not meta:
            # 没有历史统计数据，无法估算
            return False

        # 1) 计算平均一个字符花费多少token
        msg_len = meta["msg_len"]  # 历史消息字符数列表
        msg_token_len = meta["msg_token_len"]  # 历史消息token数列表
        self.avg_tokens_per_char = sum(msg_token_len) / sum(msg_len)

        # 2) 预测这次回答可能会生成多少token（基于历史平均回答长度）
        avg_char_len = sum(msg_len) / max(1, len(msg_len))
        predict_token = int(self.avg_tokens_per_char * avg_char_len)

        # 3) 判断是否可能超出最长token数量的阈值
        curr_all_token = int(predict_token + exist_chars * self.avg_tokens_per_char)
        if curr_all_token > self.config.multi_dialogue_rag.llm_max_token * self.config.multi_dialogue_rag.max_token_threshold:
            # 需要删除历史信息（压缩成摘要）
            return True
        else:
            return False

    # ---------- 上下文压缩（旧消息→摘要） ----------

    def _maybe_compress_history(self, session_id: str):
        """
        检查并可能压缩对话历史

        输入参数：
        ----------
        session_id : str
            会话ID，用于指定要检查的会话
            输入示例："user_001"

        输出：无（副作用：更新_histories和_running_summaries）

        功能说明：
        -----------
        1. 获取该会话的对话历史
        2. 估算当前历史的token消耗
        3. 如果超过阈值，调用_get_summary将旧消息压缩成摘要
        4. 支持多种token估算方法（通过estimate_token_fun配置）

        Token估算方法：
        -----------
        - avg: 使用平均值方法（_avg_estimate_over_max_token）
        - 其他: 使用ESTIMATE_FUNCTION_REGISTRY中注册的方法

        示例：
        -----------
        >>> # 假设对话已有10轮，token接近阈值
        >>> rag._maybe_compress_history("user_001")
        >>> # 如果超过阈值，前5轮会被压缩成摘要，历史被裁剪
        """
        hist = self._histories.get(session_id)

        if not hist:
            # 会话不存在或无历史，直接返回
            return

        # 估算历史消息的token消耗
        total_chars = "\n".join([m.content for m in hist.messages if hasattr(m, "content")])

        # 根据配置选择token估算方法
        if self.config.multi_dialogue_rag.estimate_token_fun != "avg":
            try:
                # 使用注册的自定义估算函数
                estimate_token = ESTIMATE_FUNCTION_REGISTRY[self.config.multi_dialogue_rag.estimate_token_fun](
                    total_chars)
                if estimate_token < self.config.multi_dialogue_rag.llm_max_token * self.config.multi_dialogue_rag.max_token_threshold:
                    return  # 未超过阈值，不需要压缩
            except Exception as e:
                logger.error("注册的估计函数错误，回退到默认avg实现...")
                print(traceback(e))
                # 回退到平均值方法
                if not self._avg_estimate_over_max_token(session_id=session_id, exist_chars=total_chars):
                    return
        elif not self._avg_estimate_over_max_token(session_id=session_id, exist_chars=total_chars):
            return  # 使用平均值方法未超过阈值

        # 超过阈值，把旧消息（按配置比例）压缩成摘要
        self._get_summary(session_id)

    def _get_summary(self, session_id: str):
        """
        将旧对话消息压缩成摘要

        输入参数：
        ----------
        session_id : str
            会话ID，用于指定要压缩的会话
            输入示例："user_001"

        输出：无（副作用：更新_running_summaries和_histories）

        功能说明：
        -----------
        1. 根据cut_dialogue_scale配置计算要压缩的消息数量
        2. 将旧消息（前cutoff条）作为历史发送给LLM生成摘要
        3. 将新生成的摘要累积到running_summaries中
        4. 从对话历史中删除已压缩的消息，保留后半段

        压缩比例：
        -----------
        keep_count = max(2, len(messages) // cut_dialogue_scale)  # 至少保留2条
        n = len(messages) - keep_count  # 要摘要的条数
        例如：10条消息，cut_dialogue_scale=2，则保留后5条，压缩前5条

        摘要累积：
        -----------
        新摘要会追加到旧摘要后面，形成累积的长期记忆

        示例：
        -----------
        >>> # 历史有10条消息
        >>> rag._get_summary("user_001")
        >>> # 前5条被压缩成摘要
        >>> # running_summaries["user_001"] = "摘要内容..."
        >>> # hist.messages 只剩下后5条
        """
        hist = self._histories.get(session_id)

        if self.config.multi_dialogue_rag.console_debug:
            logger.warning(f"[{session_id}] 对话过长，需要生成摘要...")

        # 计算要摘要的条数：至少保留2条
        keep_count = max(2, len(hist.messages) // self.config.multi_dialogue_rag.cut_dialogue_scale)
        n = len(hist.messages) - keep_count
        old_msgs = hist.messages[:n]

        # 生成摘要：构建摘要提示词并调用LLM
        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt_template("summary")["system"]),
            MessagesPlaceholder("history"),
            ("human", get_prompt_template("summary")["user"])
        ])
        summary_result: AIMessage = (summarize_prompt | self.llm).invoke({"history": old_msgs})

        # 处理摘要内容：移除思考标签等非输出内容
        summary = re.sub(r"<\|.*?\|>\s*", "", summary_result.content, flags=re.DOTALL).strip()
        dur = summary_result.response_metadata.get("total_duration", 0) / 1e9  # 转换为秒
        tokens = summary_result.usage_metadata["total_tokens"]

        if self.config.multi_dialogue_rag.console_debug:
            logger.warning(f"[{session_id}] 摘要生成完毕，耗时：{dur} s，使用tokens：{tokens}\n摘要文本：\n{summary}")

        # 累积摘要：新摘要追加到旧摘要后面
        prev = self._running_summaries.get(session_id, "")
        merged = (prev + "\n" + summary).strip() if prev else summary
        self._running_summaries[session_id] = merged

        # 丢弃已压缩的旧消息，保留后半段
        hist.messages = hist.messages[n:]

    @staticmethod
    def _strip_think_get_tokens(msg: AIMessage):
        """
        处理LLM返回消息，移除思考标签并提取token统计信息

        输入参数：
        ----------
        msg : AIMessage
            LLM返回的AIMessage对象，包含content、usage_metadata、response_metadata等
            输入示例：AIMessage(content="<|think|>思考...|>实际回答", usage_metadata={...})

        输出：
        -------
        dict
            包含处理后的消息内容和统计信息的字典：
            {
                "msg": "实际回答（已移除思考标签）",
                "msg_len": 原始消息字符数,
                "msg_token_len": 输出token数,
                "generate_time": 生成耗时（秒）
            }
            输出示例：{"msg": "建议服用感冒药...", "msg_len": 100, "msg_token_len": 50, "generate_time": 1.5}

        功能说明：
        -----------
        1. 使用正则表达式移除<|...|>格式的思考标签
        2. 提取消息的字符长度、token数、生成时间等统计信息
        3. 返回处理后的字典供后续使用

        示例：
        -----------
        >>> msg = AIMessage(
        ...     content="<|think|>需要思考|>感冒药有很多",
        ...     usage_metadata={"output_tokens": 20},
        ...     response_metadata={"total_duration": 1000000000}
        ... )
        >>> result = MultiDialogueRag._strip_think_get_tokens(msg)
        >>> print(result["msg"])  # "感冒药有很多"
        >>> print(result["msg_token_len"])  # 20
        >>> print(result["generate_time"])  # 1.0
        """
        text = msg.content
        # 用于衡量大概每一个字消耗多少token
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
            history_msgs: List[Union[AIMessage, HumanMessage, SystemMessage]]
    ) -> str:
        """
        根据token预算构建文档上下文字符串

        输入参数：
        ----------
        documents : List[Document]
            检索到的文档列表
            输入示例：[Document(page_content="文档1内容..."), Document(page_content="文档2内容...")]
        rewritten_query : str
            改写后的查询问题
            输入示例："感冒后应该注意什么？"
        session_id : str
            会话ID，用于获取该会话的运行摘要
            输入示例："user_001"
        history_msgs : List[Union[AIMessage, HumanMessage, SystemMessage]]
            历史消息列表
            输入示例：[HumanMessage("我感冒了"), AIMessage("请注意休息...")]

        输出：
        -------
        str
            构建好的文档上下文字符串，包含尽可能多的文档内容
            输出示例："## 文档1：\n文档1内容...\n## 文档2：\n文档2内容..."

        功能说明：
        -----------
        1. 计算历史、摘要、系统提示、用户问题占用的token
        2. 剩余token用于文档内容
        3. 遍历文档，逐个添加直到token预算用完
        4. 如果文档放不下，截断部分内容并标记
        5. 支持多种token估算方法

        预算分配：
        -----------
        remain_token = llm_max_token - (history + summary + system + user)
        然后根据remain_token填充文档内容

        示例：
        -----------
        >>> docs = [Document(page_content="文档1内容"), Document(page_content="文档2内容")]
        >>> context = rag._build_document_context(docs, "感冒吃什么药？", "user_001", [])
        >>> # 返回类似 "## 文档1：\n文档1内容\n## 文档2：\n文档2内容"
        """
        remain_token = self.config.multi_dialogue_rag.llm_max_token
        his_text = "\n".join(
            getattr(m, "content", "") for m in history_msgs if hasattr(m, "content")
        )

        # 计算提示词各部分占用的token（文档除外）
        user_text = get_prompt_template("dialogue_rag")["user"].format(
            llm_rewritten_content=rewritten_query,
            all_document_str=""  # 先占位，后面再计算文档长度
        )
        summaries_text = self._running_summaries[session_id]
        system_text = get_prompt_template("dialogue_rag")["system"].format(running_summary=summaries_text)
        all_chars = his_text + system_text + user_text

        parts = []
        used = 0

        # 根据配置选择token估算方法
        if self.config.multi_dialogue_rag.estimate_token_fun != "avg":
            # 使用注册的估算函数
            all_token = ESTIMATE_FUNCTION_REGISTRY[self.config.multi_dialogue_rag.estimate_token_fun](all_chars)
            # 减去 llm_max_token * 0.01 是为了 预留 1% 的 token 安全缓冲区，
            # 防止因 token 估算误差导致总 token 数超过模型限制，
            # 确保 LLM 调用不会因超出 token 预算而失败。
            remain_token -= all_token + self.config.multi_dialogue_rag.llm_max_token * 0.01
        else:
            # 使用平均值方法
            all_prompt_chars_len = len(all_chars)
            remain_token -= all_prompt_chars_len + self.config.multi_dialogue_rag.llm_max_token * 0.01

        # 遍历文档，逐个添加直到token预算用完
        for idx, d in enumerate(documents):
            header = f"## 文档{idx + 1}：\n"
            body = d.page_content or ""

            # 估算文档标题和内容的token数
            if self.config.multi_dialogue_rag.estimate_token_fun != "avg":
                header_tokens = ESTIMATE_FUNCTION_REGISTRY[self.config.multi_dialogue_rag.estimate_token_fun](header)
                body_tokens = ESTIMATE_FUNCTION_REGISTRY[self.config.multi_dialogue_rag.estimate_token_fun](body)
            else:
                header_tokens = self.avg_tokens_per_char * len(header)
                body_tokens = self.avg_tokens_per_char * len(body)

            if used + header_tokens + body_tokens <= remain_token:
                # 放得下整个文档，完整添加
                parts.append(header + body + "\n")
                used += header_tokens + body_tokens
            else:
                # 放不下整篇文档，需要截断
                if self.config.multi_dialogue_rag.console_debug:
                    logger.warning(
                        f"根据给定的token估计方法，预估无法完成全部文档编码，文档{idx + 1}被截断，后续文档将无法被放入上下文...")

                # 计算剩余可用的token
                remain = remain_token - used - header_tokens
                if remain > 0:
                    # 依据平均token估算可保留字符数
                    keep_chars = max(0, int(remain / max(0.1, self.avg_tokens_per_char)))
                    if keep_chars > 0:
                        # 截断文档内容并添加标记
                        parts.append(header + body[:keep_chars] + "\n...[内容已截断]\n")
                        used = remain_token  # 填满预算
                break  # 无论是否部分放入，预算已到

        return "".join(parts)

    # ---------- 构建多轮 RAG 链 ----------

    def _setup_chain(self):
        """
        构建完整的多轮对话RAG处理链

        输入：无

        输出：无（副作用：设置self.rag_chain）

        功能说明：
        -----------
        构建一个LangChain处理链，包含以下步骤：
        1. 查询改写链：根据历史将用户问题改写为独立查询
        2. 检索链：使用改写后的问题检索相关文档
        3. 格式化链：构建文档上下文
        4. 生成链：基于上下文生成最终答案

        处理流程：
        -----------
        original_input
          -> rewritten_query_chain (改写问题)
          -> do_retrieve (检索文档)
          -> do_format (格式化文档)
          -> dialogue_rag_prompt + llm (生成答案)
          -> strip_think (处理输出)

        示例：
        -----------
        >>> rag._setup_chain()
        >>> result = rag.rag_chain.invoke(
        ...     {"original_input": "感冒吃什么？", "session_id": "user_001"},
        ...     config={"configurable": {"session_id": "user_001"}}
        ... )
        >>> print(result["answer"])
        """
        # 构建查询改写链：history + user_input -> rewritten_query
        rewrite_template = ChatPromptTemplate.from_messages([
            ("system", get_prompt_template("rewriter")["system"]),
            MessagesPlaceholder("history"),
            ("human", get_prompt_template("rewriter")["user"])
        ])
        # 填充模板 -> llm生成 -> 处理think标签
        rewritten_query_chain = rewrite_template | self.llm | RunnableLambda(self._strip_think_get_tokens)

        def do_retrieve(inputs: dict):
            """使用改写后的问题进行检索
            inputs = {
                    # 原始用户输入（来自 invoke 调用）
                    "original_input": "感冒吃什么？",

                    # 会话ID
                    "session_id": "user_001",

                    # 对话历史消息（由 RunnableWithMessageHistory 自动填充）
                    "history": [
                        HumanMessage("我感冒了"),
                        AIMessage("请注意休息...")
                    ],

                    # 改写后的查询（由 rewritten_query_chain 生成并赋值）
                    "llm_rewritten_query": {
                        "msg": "感冒后应该服用哪些药物治疗？",  # 改写后的问题
                        "msg_len": 25,                          # 原始消息字符数
                        "msg_token_len": 50,                    # 输出 token 数
                        "generate_time": 1.2                    # 生成耗时（秒）
                    }
                }
            """
            logger.info(f"改写后的问题: {inputs['llm_rewritten_query']['msg']}")
            return self.self_retriever.invoke({"input": inputs["llm_rewritten_query"]["msg"]})

        def do_format(inputs: dict) -> str:
            """格式化文档上下文
            inputs = {
                # 原始用户输入
                "original_input": "感冒吃什么？",

                # 会话ID
                "session_id": "user_001",

                # 对话历史（由 RunnableWithMessageHistory 自动填充）
                "history": [
                    HumanMessage("之前的用户问题"),
                    AIMessage("之前的AI回答")
                ],

                # 改写后的查询（由 rewritten_query_chain 生成）
                "llm_rewritten_query": {
                    "msg": "感冒后应该服用哪些药物治疗？",
                    "msg_len": 25,
                    "msg_token_len": 50,
                    "generate_time": 1.2
                },

                # 检索结果（由 do_retrieve 返回并添加到 inputs 中）
                "milvus_result": {
                    "documents": [  # 检索到的文档列表
                        Document(page_content="文档1内容...", metadata={...}),
                        Document(page_content="文档2内容...", metadata={...})
                    ],
                    "search_time": 0.5,  # 检索耗时（秒）
                    # 可能有其他检索相关的元数据
                }
            }
            """
            documents: List[Document] = inputs["milvus_result"]["documents"]
            all_document_str = self._build_document_context(
                documents=documents,
                rewritten_query=inputs["llm_rewritten_query"]["msg"],
                session_id=inputs.get("session_id", "default"),
                history_msgs=inputs.get("history", [])
            )
            return {**inputs, "all_document_str": all_document_str,
                    "llm_rewritten_content": inputs["llm_rewritten_query"]["msg"]}

        # 构建答案生成链：格式化 -> 提示词 -> LLM -> 处理输出
        out_answer = (
                RunnableLambda(do_format)
                | self.dialogue_rag_prompt
                | self.llm
                | RunnableLambda(self._strip_think_get_tokens)
        )

        # 构建核心链：逐步执行各个步骤
        core_chain = (
                RunnablePassthrough.assign(llm_rewritten_query=rewritten_query_chain)
                .with_config(run_name="rewritten_query")
                | RunnablePassthrough.assign(milvus_result=RunnableLambda(do_retrieve))
                .with_config(run_name="search_documents")
                | RunnablePassthrough.assign(llm_out_result=out_answer)
                .with_config(run_name="generate")
                | RunnableLambda(lambda x: {**x, "answer": x["llm_out_result"]["msg"]})  # 提取最终答案
        )

        def _get_history_wrapper(session_id: str):
            """包装历史获取器供RunnableWithMessageHistory使用"""
            return self._get_history(session_id)

        # 包装成支持历史管理的链
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
        """
        更新token元数据，用于更准确的token估算

        输入参数：
        ----------
        answer_result : dict
            RAG处理结果，包含改写和生成的token信息
            输入示例：{
                "llm_rewritten_query": {"msg_len": 50, "msg_token_len": 25},
                "llm_out_result": {"msg_len": 200, "msg_token_len": 100}
            }
        session_id : str
            会话ID
            输入示例："user_001"

        输出：无（副作用：更新self._token_meta_store）

        功能说明：
        -----------
        1. 将本次对话的改写和生成的字符数、token数记录下来
        2. 用于后续更准确地估算token消耗
        3. 每次交互都会累积更多数据，估算越来越准确

        数据结构：
        -----------
        _token_meta_store[session_id] = {
            "msg_len": [字符数1, 字符数2, ...],
            "msg_token_len": [token数1, token数2, ...]
        }

        示例：
        -----------
        >>> result = {
        ...     "llm_rewritten_query": {"msg_len": 50, "msg_token_len": 25},
        ...     "llm_out_result": {"msg_len": 200, "msg_token_len": 100}
        ... }
        >>> rag._update_tokens_metadata(result, "user_001")
        >>> # _token_meta_store["user_001"]["msg_len"] 变成 [50, 200]
        """
        # 更新token信息，以便估算下一次对话是否需要摘要
        # {"session_id": {"msg_len": List[int], "msg_token_len": List[int]}}
        if self._token_meta_store.get(session_id) is not None:
            # 已有数据，追加新的记录
            self._token_meta_store[session_id]["msg_len"].append(answer_result["llm_rewritten_query"]["msg_len"])
            self._token_meta_store[session_id]["msg_token_len"].append(
                answer_result["llm_rewritten_query"]["msg_token_len"])
            self._token_meta_store[session_id]["msg_len"].append(answer_result["llm_out_result"]["msg_len"])
            self._token_meta_store[session_id]["msg_token_len"].append(answer_result["llm_out_result"]["msg_token_len"])
        else:
            # 首次记录，初始化列表
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

    # ---------- 对外 API：增加 session_id & 多轮 ----------

    def answer(
            self,
            query: str,
            return_document: bool = False,
            session_id: str = "default"
    ) -> Union[str, Dict[str, Union[str, List[Document]]]]:
        """
        处理用户问题并返回答案（对外主API）

        输入参数：
        ----------
        query : str
            用户问题
            输入示例："感冒吃什么药？"
        return_document : bool, 可选
            是否返回检索到的文档，默认False
            输入示例：True
        session_id : str, 可选
            会话ID，用于区分不同用户会话，默认"default"
            输入示例："user_001"

        输出：
        -------
        Union[str, Dict[str, Union[str, List[Document]]]]
            - return_document=False: 返回答案字符串或字典（带时间统计）
            - return_document=True: 返回完整信息字典
            输出示例（return_document=False）：
                {
                    "answer": "建议服用感冒灵颗粒...",
                    "search_time": 0.5,
                    "rewriten_generate_time": 1.2,
                    "out_generate_time": 3.5
                }
            输出示例（return_document=True）：
                {
                    "answer": "建议服用感冒灵颗粒...",
                    "documents": [Document(...), Document(...)],
                    "search_time": 0.5,
                    "rewriten_generate_time": 1.2,
                    "out_generate_time": 3.5
                }

        功能说明：
        -----------
        1. 检查并可能压缩历史（释放token空间）
        2. 调用RAG链处理用户问题（改写->检索->生成）
        3. 更新token统计（用于后续更准确的估算）
        4. 根据return_document参数返回相应结果
        5. 异常处理：出错时返回友好的错误信息

        处理流程：
        -----------
        query
          -> _maybe_compress_history (可能压缩历史)
          -> rag_chain.invoke (执行RAG处理)
          -> _update_tokens_metadata (更新统计)
          -> return result

        示例：
        -----------
        >>> # 简单查询
        >>> result = rag.answer("感冒吃什么药？", session_id="user_001")
        >>> print(result["answer"])

        >>> # 获取检索到的文档
        >>> result = rag.answer("感冒吃什么药？", return_document=True, session_id="user_001")
        >>> print(result["answer"])
        >>> for doc in result["documents"]:
        ...     print(doc.page_content)
        """
        logger.info(f"[{session_id}] 问题: {query}")
        try:
            # 1) 可能压缩旧历史（释放token空间）
            self._maybe_compress_history(session_id)

            # 2) 运行RAG处理链（注意 config 中要传 session_id）
            result = self.rag_chain.invoke(
                {
                    "original_input": query,
                    "running_summary": self._running_summaries.get(session_id, ""),
                    "session_id": session_id
                },
                config={"configurable": {"session_id": session_id}}
            )

            # 3) 更新token计数，为估计更准确的token数以生成摘要
            self._update_tokens_metadata(answer_result=result, session_id=session_id)

            # 提取答案，提供默认值
            answer = result.get("answer", "抱歉，根据提供的资料无法回答您的问题。")

            # 根据参数返回不同格式
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

            # 出错时返回友好的错误信息
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

    # ---------- 更新检索配置 ----------

    def update_search_config(self, search_config: SearchRequest):
        """
        更新检索配置并重建RAG链

        输入参数：
        ----------
        search_config : SearchRequest
            新的检索配置对象
            输入示例：SearchRequest(top_k=10, hybrid_alpha=0.7)

        输出：无（副作用：更新self.self_retriever和self.rag_chain）

        功能说明：
        -----------
        1. 使用新配置创建新的混合检索器
        2. 重新构建RAG处理链（因为检索器已改变）
        3. 记录日志

        使用场景：
        -----------
        - 动态调整检索参数（如top_k、hybrid_alpha）
        - 根据用户反馈优化检索策略

        示例：
        -----------
        >>> new_config = SearchRequest(top_k=10, hybrid_alpha=0.8)
        >>> rag.update_search_config(new_config)
        >>> # 后续检索会使用新配置
        """
        self.self_retriever = MedicalHybridRetriever(self.knowledge_base, search_config)
        self._setup_chain()
        logger.info(f"搜索配置已更新: {search_config}")
