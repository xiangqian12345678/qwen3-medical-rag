import os

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
文档分块检索后，与查询最相关的信息可能隐藏在一个包含大量不相关文本的文档中，输入给LLM，可能会导致更昂贵的LLM调用和较差的响应（噪声）。
"""


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# 1.数据准备
RESOURCE_DIR = "../../data/base/resources"
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

print("\n1. 压缩前" + '-' * 100)
pretty_print_docs(docs)

# 4.压缩过滤
llm = ChatTongyi(model="qwen-max")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
compressed_docs = compression_retriever.invoke("deepseek的发展历程")

print("\n2. 压缩后" + '-' * 100)
pretty_print_docs(compressed_docs)
