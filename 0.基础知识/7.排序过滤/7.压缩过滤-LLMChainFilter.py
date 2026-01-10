import os

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_community.chat_models.tongyi import ChatTongyi
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
'''
Given the following question and context, return YES if the context is relevant to the question and NO if it isn't.
> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):

给定以下问题和上下文，如果上下文与问题相关，则返回YES，否则返回NO。
问题：{question}
上下文：
>>>
{context}
>>>
相关性（是/否）：
'''
llm = ChatTongyi(model="qwen-max")
_filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter,
    base_retriever=retriever
)

# 5. 过滤召回
compressed_docs = compression_retriever.invoke("deepseek的发展历程")

print("\n2.LLMChainFilter过滤后" + '-' * 100)
pretty_print_docs(compressed_docs)
