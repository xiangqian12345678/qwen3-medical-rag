"""工具类"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.models import AppConfig, LLMConfig, DenseFieldConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Token估算函数注册表
# =============================================================================
ESTIMATE_FUNCTION_REGISTRY: Dict[str, callable] = {}


def register_estimate_function(name: str):
    """注册自定义token估算函数的装饰器"""
    def decorator(func: callable):
        ESTIMATE_FUNCTION_REGISTRY[name] = func
        return func
    return decorator


@register_estimate_function("tiktoken")
def estimate_tokens_tiktoken(text: str) -> int:
    """
    使用tiktoken库估算token数

    Args:
        text: 输入文本

    Returns:
        token数量
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        logger.warning("tiktoken未安装，使用字符数估算")
        return len(text)


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
        if config.proxy:
            kwargs["http_client"] = {"proxies": {"http": config.proxy, "https": config.proxy}}
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
        documents: 文档列表

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
