from rag.rag_config import RerankerConfig
from utils import create_reranker_client

if __name__ == '__main__':
    config = RerankerConfig(
        provider="ollama",
        model="dengcao/bge-reranker-v2-m3",
        base_url="http://localhost:11434"
    )

    reranker = create_reranker_client(config)

    query = "什么是倒排索引？"
    documents = [
        "倒排索引用于从词找到文档。",
        "Transformer 是一种神经网络结构。"
    ]

    scores = reranker.rerank(query=query, documents=documents)
    print(scores)