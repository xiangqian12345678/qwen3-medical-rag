"""工具类"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.models import AppConfig, LLMConfig, DenseFieldConfig


logger = logging.getLogger(__name__)


def create_llm_client(config: LLMConfig) -> BaseChatModel:
    """创建LLM客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
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
        return ChatOllama(**kwargs)

    else:
        raise ValueError(f"不支持的LLM提供商: {config.provider}")


def create_embedding_client(config: DenseFieldConfig) -> Embeddings:
    """创建嵌入客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model,
            "dimensions": config.dimension
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OpenAIEmbeddings(**kwargs)

    elif config.provider == "ollama":
        kwargs = {"model": config.model}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OllamaEmbeddings(**kwargs)

    else:
        raise ValueError(f"不支持的嵌入提供商: {config.provider}")


def format_documents(documents: List[Any]) -> str:
    """
    格式化文档为字符串

    Args:
        documents: 文档列表，元素类型为 Document 或 dict
            Document 示例结构:
            Document(
                page_content="糖尿病是一组以高血糖为特征的代谢性疾病...",
                metadata={
                    "pk": "primary_key",
                    "distance": 0.123,
                    "chunk": "糖尿病是一组以高血糖为特征的代谢性疾病...",
                    "parent_chunk": "更长的父块内容...",
                    "summary": "摘要内容...",
                    "questions": "相关问题...",
                    "source": "来源...",
                    "source_name": "来源名称...",
                    "lt_doc_id": "doc_123",
                    "chunk_id": 0,
                    "hash_id": "abc123"
                }
            )

    Returns:
        格式化后的文档字符串
    """
    parts = []
    for i, doc in enumerate(documents):
        if hasattr(doc, 'page_content'):
            content = doc.page_content
        else:
            content = str(doc)
        parts.append(f"## 文档{i + 1}：\n{content}\n")
    return "".join(parts)
