"""
嵌入向量服务模块
负责调用大模型API生成文本嵌入向量
"""
from typing import List
import httpx
from config import config


class EmbeddingService:
    """
    嵌入向量生成服务类
    使用通义千问API生成文本的嵌入向量
    """

    def __init__(self, api_key: str = None):
        """
        初始化嵌入服务

        【输入示例】
        service = EmbeddingService(api_key="sk-xxx")

        【输出示例】
        None (服务已初始化)
        """
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        self.api_url = config.EMBEDDING_URL
        self.model = config.EMBEDDING_MODEL
        self.cache = {}  # 嵌入缓存

        # 创建HTTP客户端
        self.client = httpx.Client(
            timeout=60.0,
            trust_env=False
        )

    def generate_embedding(self, text: str) -> List[float]:
        """
        为单个文本生成嵌入向量

        【输入示例】
        text = "药物: 阿司匹林"
        embedding = service.generate_embedding(text)

        【输出示例】
        [0.0234, -0.0123, 0.0456, ..., 0.0189]  # 1536维向量
        """
        if not text:
            return []

        # 检查缓存
        if text in self.cache:
            return self.cache[text]

        try:
            response = self.client.post(
                self.api_url,
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
                self.cache[text] = embedding
                return embedding

            print(f"API响应格式错误: {data}")
            return []
        except Exception as e:
            print(f"生成嵌入失败: {e}")
            return []

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入向量

        【输入示例】
        texts = ["药物: 阿司匹林", "药物: 布洛芬", "症状: 头痛"]
        embeddings = service.generate_embeddings_batch(texts)

        【输出示例】
        [
            [0.0234, -0.0123, 0.0456, ...],  # 阿司匹林的嵌入
            [0.0156, 0.0321, -0.0089, ...],  # 布洛芬的嵌入
            [0.0421, -0.0198, 0.0076, ...]   # 头痛的嵌入
        ]
        """
        if not texts:
            return []

        embeddings = []
        uncached_texts = []

        # 分离已缓存和未缓存的文本
        for text in texts:
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                uncached_texts.append(text)

        # 为未缓存的文本生成嵌入
        if uncached_texts:
            batch_size = 10
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]

                try:
                    response = self.client.post(
                        self.api_url,
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
                        for j, embedding_data in enumerate(data["output"]["embeddings"]):
                            embedding = embedding_data["embedding"]
                            text = batch[j]
                            self.cache[text] = embedding
                            embeddings.append(embedding)

                except Exception as e:
                    print(f"批量生成嵌入失败: {e}")
                    for _ in batch:
                        embeddings.append([])

        return embeddings

    def clear_cache(self):
        """
        清空嵌入缓存

        【输入示例】
        service.clear_cache()

        【输出示例】
        None (缓存已清空)
        """
        self.cache.clear()
        print("🧹 嵌入缓存已清空")

    def close(self):
        """
        关闭HTTP客户端

        【输入示例】
        service.close()

        【输出示例】
        None (客户端已关闭)
        """
        self.client.close()


# 使用示例
if __name__ == "__main__":
    # 示例1: 单个文本嵌入
    print("示例1: 单个文本嵌入")
    service = EmbeddingService()
    text = "药物: 阿司匹林"
    embedding = service.generate_embedding(text)
    print(f"文本: {text}")
    print(f"嵌入维度: {len(embedding)}")

    # 示例2: 批量嵌入
    print("\n示例2: 批量嵌入")
    texts = [
        "药物: 阿司匹林",
        "药物: 布洛芬",
        "症状: 头痛",
        "疾病: 感冒"
    ]
    embeddings = service.generate_embeddings_batch(texts)
    for i, (t, emb) in enumerate(zip(texts, embeddings)):
        print(f"{i+1}. {t} -> 维度: {len(emb)}")

    # 示例3: 缓存测试
    print("\n示例3: 缓存测试")
    text = "测试文本"
    print("第一次生成...")
    _ = service.generate_embedding(text)
    print("第二次生成(从缓存)...")
    _ = service.generate_embedding(text)
    print(f"缓存大小: {len(service.cache)}")

    service.close()
