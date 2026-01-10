import os

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# 1.文件路径
RESOURCE_DIR = "/mnt/c/大模型/智泊大模型全栈教程总结/02-教材整理 L2/代码/Langchain/6.langchain高级RAG/data/resources"
TXT_DOCUMENT_PATH = os.path.join(RESOURCE_DIR, "deepseek百度百科.txt")

# 2.模型准备
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

# 3.数据加载
loader = TextLoader(TXT_DOCUMENT_PATH, encoding='utf-8')
docs = loader.load()

# 4.文档向量化并存储
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings_model, collection_name="mix"
)

# 5.文档召回
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
query = "相关评价"
vector_retriever_doc = vector_retriever.invoke(query)
print("\n\n向量召回文档：" + "==" * 100)
pretty_print_docs(vector_retriever_doc)

# 6.BM25检索
BM25_retriever = BM25Retriever.from_documents(chunks, k=3)
BM25Retriever_doc = BM25_retriever.invoke(query)
print("\n\n关键词召回文档：" + "==" * 100)
pretty_print_docs(BM25Retriever_doc)

# 7.混合检索
# 向量检索和关键词检索的权重各0.5，两者赋予相同的权重
retriever = EnsembleRetriever(retrievers=[BM25_retriever, vector_retriever], weights=[0.5, 0.5])
print("\n\n混合召回文档：" + "==" * 100)
pretty_print_docs(retriever.invoke(query))
