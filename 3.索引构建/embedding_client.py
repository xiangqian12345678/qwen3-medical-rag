"""嵌入模型客户端工厂"""
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from config import DenseFieldConfig


def create_embedding_client(config: DenseFieldConfig) -> Embeddings:
    """创建嵌入模型客户端

    Args:
        config: 稠密向量字段配置

    Returns:
        Embeddings: 嵌入模型实例

    Raises:
        ValueError: 不支持的嵌入提供商
    """
    if config.provider == "openai":
        kwargs = {"model": config.model, "dimensions": config.dimension}
        # OpenAI API 可能需要 api_key
        # if config.api_key:
        #     kwargs["api_key"] = config.api_key
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
