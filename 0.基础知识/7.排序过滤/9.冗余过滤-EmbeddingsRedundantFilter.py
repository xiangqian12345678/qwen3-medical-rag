import os

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter
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
# 默认文档间相似度超过0.95则为冗余文档
# 计算所有文档之间的嵌入向量相似度
# 对于相似度超过阈值的文档对，只保留其中一个
# 直接使用  redundant_filter.transform_documents(documents)
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model, similarity_threshold=0.95)
# 根据问题与文档的相似度过滤
relevant_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.66)

# 首先应用redundant_filter去除冗余文档
# 然后应用relevant_filter去除与查询不相关的文档
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, relevant_filter]
)

# 5. 过滤召回
# 首先通过基础检索器(Chroma)获取初步检索结果
# 然后通过压缩管道对结果进行过滤和优化
# 最终返回精炼后的、高质量的文档集
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)
compressed_docs = compression_retriever.invoke("deepseek的发展历程")

print("\n2.过滤后" + '-' * 100)
pretty_print_docs(compressed_docs)
