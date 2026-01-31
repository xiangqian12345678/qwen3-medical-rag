"""工具函数"""
import logging

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
