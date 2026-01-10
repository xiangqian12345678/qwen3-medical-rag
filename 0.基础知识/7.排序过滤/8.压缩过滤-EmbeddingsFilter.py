import os

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# 1.数据准备
RESOURCE_DIR = "/mnt/c/大模型/智泊大模型全栈教程总结/02-教材整理 L2/代码/Langchain/6.langchain高级RAG/data/resources"
TXT_DOCUMENT_PATH = os.path.join(RESOURCE_DIR, "deepseek百度百科.txt")
loader = TextLoader(TXT_DOCUMENT_PATH, encoding='utf-8')
documents = loader.load()

# 2.向量化并索引
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")
retriever = Chroma.from_documents(chunks, embeddings_model).as_retriever()

# 3.测试召回
docs = retriever.invoke("deepseek的发展历程")
print("\n1.向量召回" + '-' * 100)
pretty_print_docs(docs)

# 4. 过滤器
# 对检索到的文档块与查询进行相似度计算，如果相似度大于0.66，则保留该文档块，否则过滤掉
embeddings_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.66)

# 5. 过滤召回
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("deepseek的发展历程")

print("\n2.过滤后" + '-' * 100)
pretty_print_docs(compressed_docs)
