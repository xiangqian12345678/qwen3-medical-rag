"""嵌入模型客户端工厂"""
from typing import List, Dict, Any, Optional
import httpx
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from embed_config import DenseFieldConfig


class DashScopeEmbeddings(Embeddings):
    """DashScope 嵌入模型客户端，兼容 LangChain Embeddings 接口"""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v2",
        base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        dimensions: Optional[int] = None
    ):
        """
        初始化 DashScope 嵌入客户端

        Args:
            api_key: DashScope API 密钥
            model: 嵌入模型名称
            base_url: API 基础 URL
            dimensions: 嵌入向量维度（可选）
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.dimensions = dimensions
        self.client = httpx.Client(timeout=60.0, trust_env=False)

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本

        Args:
            text: 待嵌入的文本

        Returns:
            List[float]: 嵌入向量
        """
        try:
            response = self.client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": {"texts": [text]}
                }
            )
            response.raise_for_status()
            data = response.json()

            if "output" in data and "embeddings" in data["output"]:
                embedding = data["output"]["embeddings"][0]["embedding"]
                return embedding

            raise ValueError(f"API 响应格式错误: {data}")
        except Exception as e:
            raise RuntimeError(f"DashScope 嵌入失败: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档文本

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        embeddings = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = self.client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "input": {"texts": batch}
                    }
                )
                response.raise_for_status()
                data = response.json()

                if "output" in data and "embeddings" in data["output"]:
                    for embedding_data in data["output"]["embeddings"]:
                        embedding = embedding_data["embedding"]
                        embeddings.append(embedding)
                else:
                    raise ValueError(f"API 响应格式错误: {data}")

            except Exception as e:
                raise RuntimeError(f"DashScope 批量嵌入失败: {e}")

        return embeddings

    def close(self):
        """关闭 HTTP 客户端"""
        self.client.close()


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
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return OpenAIEmbeddings(**kwargs)

    elif config.provider == "ollama":
        kwargs = {"model": config.model}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OllamaEmbeddings(**kwargs)

    elif config.provider == "dashscope":
        # Dashscope 需要 api_key
        if not config.api_key:
            raise ValueError("Dashscope provider 需要 api_key 配置")
        return DashScopeEmbeddings(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            dimensions=config.dimension
        )

    else:
        raise ValueError(f"不支持的嵌入提供商: {config.provider}")
