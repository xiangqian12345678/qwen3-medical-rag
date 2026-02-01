"""工具函数"""
import logging
import time
from typing import Any, Callable, Optional

import httpx
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from rag.rag_config import LLMConfig, EmbeddingConfig, RerankerConfig

logger = logging.getLogger(__name__)


def create_llm_client(config: LLMConfig) -> BaseChatModel:
    """创建LLM客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        return ChatOpenAI(**kwargs)

    elif config.provider == "ollama":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.max_tokens:
            kwargs["num_predict"] = config.max_tokens
        logger.info(f"创建 Ollama LLM: {config.model}, base_url: {config.base_url}")
        return ChatOllama(**kwargs)

    elif config.provider == "dashscope":
        # 创建 HTTP 客户端，参考知识图谱模块的配置
        http_client = httpx.Client(
            trust_env=False,
            timeout=60.0
        )

        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
            "http_client": http_client
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        logger.info(f"创建 DashScope LLM: {config.model}")
        return ChatTongyi(**kwargs)

    else:
        raise ValueError(f"不支持的LLM提供商: {config.provider}")


def create_embedding_client(config: EmbeddingConfig) -> Embeddings:
    """创建Embedding客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OpenAIEmbeddings(**kwargs)

    elif config.provider == "ollama":
        kwargs = {
            "model": config.model,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        logger.info(f"创建 Ollama Embedding: {config.model}, base_url: {config.base_url}")
        return OllamaEmbeddings(**kwargs)

    elif config.provider == "dashscope":
        kwargs = {
            "model": config.model,
        }
        if config.api_key:
            kwargs["dashscope_api_key"] = config.api_key
        logger.info(f"创建 DashScope Embedding: {config.model}")
        return DashScopeEmbeddings(**kwargs)

    else:
        raise ValueError(f"不支持的Embedding提供商: {config.provider}")


def create_reranker_client(config: RerankerConfig):
    """
    创建 Reranker 客户端，支持 dashscope 和 ollama
    """
    if config.provider == "dashscope":
        try:
            from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
        except ImportError as e:
            raise ImportError("请安装 langchain-community 并包含 dashscope_rerank 模块") from e

        kwargs = {"model": config.model}
        if config.api_key:
            kwargs["dashscope_api_key"] = config.api_key

        logger.info(f"创建 DashScope Reranker: {config.model}")
        return DashScopeRerank(**kwargs)

    elif config.provider == "ollama":
        try:
            import requests
        except ImportError:
            raise ImportError("请安装 requests 库以支持 Ollama Reranker 调用")

        class OllamaReranker:
            """简单封装 Ollama rerank API"""

            def __init__(self, model: str, base_url: str = "http://localhost:11434"):
                self.model = model
                self.base_url = base_url.rstrip("/")

            def rerank(self, query: str, documents: list):
                url = f"{self.base_url}/v1/rerank"
                payload = {
                    "model": self.model,
                    "query": query,
                    "documents": documents
                }
                resp = requests.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()["results"]

        logger.info(f"创建 Ollama Reranker: {config.model}, Base URL: {config.base_url}")
        return OllamaReranker(model=config.model, base_url=config.base_url)

    else:
        raise ValueError(f"不支持的 Reranker 提供商: {config.provider}")


def invoke_with_timing(
    func: Callable,
    *args,
    stage_name: Optional[str] = None,
    state: Optional[dict] = None,
    **kwargs
) -> Any:
    """
    带计时和性能日志的通用函数调用包装器

    Args:
        func: 要调用的函数
        *args: 函数的位置参数
        stage_name: 阶段名称（用于日志记录和性能记录），默认为函数名
        state: Agent状态对象，如果提供会将性能信息记录到 state["performance"]
        **kwargs: 函数的关键字参数

    Returns:
        函数的执行结果

    Example:
        # 简单使用
        result = invoke_with_timing(
            func=_parallel_recall,
            query_list=sub_queries.queries,
            max_parallel=cpu_count,
            stage_name="parallel_recall"
        )

        # 记录到state
        result = invoke_with_timing(
            func=llm_chain.invoke,
            inputs={"query": query},
            stage_name="llm_call",
            state=agent_state
        )
    """
    # 确定阶段名称
    if stage_name is None:
        stage_name = func.__name__

    # 记录开始时间
    start_time = time.time()

    # 执行函数
    result = func(*args, **kwargs)

    # 计算耗时
    elapsed_time = time.time() - start_time

    # 记录性能日志
    logger.info(f"  {stage_name}: {elapsed_time:.2f}秒")

    # 如果提供了state，将性能信息记录到performance列表
    if state is not None and "performance" in state:
        perf_info = {
            "stage": stage_name,
            "duration": elapsed_time,
            "timestamp": time.time()
        }
        state["performance"].append(perf_info)

    return result
