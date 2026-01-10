import logging
from typing import List
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# 1. 模型初始化
llm = ChatTongyi(model="qwen-max")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

# 2. 准备数据
texts = [
    "人工智能在医疗诊断中的应用。",
    "人工智能如何提升供应链效率。",
    "NBA季后赛最新赛况分析。",
    "传统法式烘焙的五大技巧。",
    "红楼梦人物关系图谱分析。",
    "人工智能在金融风险管理中的应用。",
    "人工智能如何影响未来就业市场。",
    "人工智能在制造业的应用。",
    "今天天气怎么样",
    "人工智能伦理：公平性与透明度。"
]

# 为每个文档生成唯一ID和元数据
ids = [str(uuid4()) for _ in range(len(texts))]  # 生成UUID作为ID
metadatas = [{"source": f"doc_{i + 1}", "id": f"{ids[i]}"} for i in range(len(texts))]  # 可选的元数据

# 3.向量存储
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings_model,
    ids=ids,
    metadatas=metadatas,
    collection_name="rrf"
)
retriever = vectorstore.as_retriever()

# 4. 多query查询召回器重构
# 重写MultiQueryRetriever类，取消unique_union去重，且保留每个问题检索结果的
class RRFMultiQueryRetriever(MultiQueryRetriever):
    # 改写retrieve_documents方法，返回rrf结果
    def retrieve_documents(
            self, queries: List[str], run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        documents = []
        for query in queries:
            docs = self.retriever.invoke(
                query, config={"callbacks": run_manager.get_child()}
            )
            # 原代码中extend修改为append，保持不同检索系统的结构
            documents.append(docs)

        documents = self.rrf_documents(documents)
        return documents

    def rrf_documents(self, documents: list[list[Document]], k=60) -> List[Document]:
        # 初始化rrf字典（key=文档id，value={"rrf_score":累计分数,"doc":文档对象}）
        rrf_scores = {}
        # 遍历每个检索结果列表（每个查询对应的结果）
        for docs in documents:
            # 为每个文档列表计算排名（从1开始）
            for rank, doc in enumerate(docs, 1):
                # 计算当前文档的RRF分数
                rrf_score = 1 / (k + rank)
                # 如果文档已经在字典中，累加RRF分数
                if doc.metadata.get("id") in rrf_scores:
                    rrf_scores[doc.metadata.get("id")]['rrf_score'] += rrf_score
                else:
                    rrf_scores[doc.metadata.get("id")] = {'rrf_score': rrf_score, 'doc': doc}

        # 将字典转换为列表，并根据字段value：RRF分数排序
        sorted_docs = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True  # 降序排列：从大到小
        )

        result = [item['doc'] for item in sorted_docs]

        return result


# 5. 检索
rrf_retriever = RRFMultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
    include_original=True  # 是否包含原始查询
)

rrf_docs = rrf_retriever.invoke("人工智能的应用")

pretty_print_docs(rrf_docs)
