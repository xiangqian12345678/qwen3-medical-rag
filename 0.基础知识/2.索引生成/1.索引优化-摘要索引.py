import os
import uuid

from langchain_classic.retrievers import MultiVectorRetriever
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy import dot
from numpy.linalg import norm

# 初始化llm（通义千问）
llm = ChatTongyi(model="qwen-max")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")


def test_no_summary():
    '''未使用摘要索引'''
    docs = [
        "DeepSeek，全称杭州深度求索人工智能基础技术研究有限公司，是一家创新型科技公司 ，成立于2023年7月17日，使用数据蒸馏技术 ，得到更为精炼、有用的数据 。由知名私募巨头幻方量化孕育而生，专注于开发先进的大语言模型（LLM）和相关技术。",
        "DeepSeek R1是推理模型，遵循 MIT License，通过设置 model='deepseek-reasoner' 即可调用。DeepSeek V3是通用大模型，目前版本号 DeepSeek-V3-0324。通过指定 model='deepseek-chat' 即可调用 DeepSeek V3。",
    ]

    query = "DeepSeek R1模型的开发公司叫什么？"

    # 将文档和查询转换为向量
    doc_embeddings = embeddings_model.embed_documents(docs)
    query_embedding = embeddings_model.embed_query(query)

    # 计算相似度  余弦相似度：点积（A*B）/模长乘积（A模长*B模长）
    similarities = [dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding)) for
                    doc_embedding in doc_embeddings]

    print("==========================不使用摘要索引======================================")
    # 期望得到文档1,检索出文档2
    for i, similarity in enumerate(similarities):
        print(f"第{i + 1}个相似度：", similarity)


def test_summary():
    '''使用摘要索引'''
    # 假设生成了文档的摘要
    summary_docs = [
        "DeepSeek公司介绍",
        "DeepSeek模型调用说明"
    ]
    query = "DeepSeek R1模型的开发公司叫什么？"

    summary_doc_embeddings = embeddings_model.embed_documents(summary_docs)
    query_embedding = embeddings_model.embed_query(query)

    # 计算问题与文档摘要的相似度（基于摘要的检索）
    similarities = [dot(query_embedding, summary_doc_embedding) / (norm(query_embedding) * norm(summary_doc_embedding))
                    for summary_doc_embedding in summary_doc_embeddings]

    print("==========================使用摘要索引相似度======================================")
    for i, similarity in enumerate(similarities):
        print(f"第{i + 1}个相似度：", similarity)


def test_summary_index():
    # 1.文件路径
    RESOURCE_DIR = "/mnt/c/大模型/智泊大模型全栈教程总结/02-教材整理 L2/代码/Langchain/6.langchain高级RAG/data/resources"
    TXT_DOCUMENT_PATH = os.path.join(RESOURCE_DIR, "deepseek百度百科.txt")

    # 2.初始化模型
    llm = ChatTongyi(model="qwen-max")
    embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

    # 3. 加载本地文件
    loader = TextLoader(TXT_DOCUMENT_PATH, encoding='utf-8')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        print(f"块 {i + 1} :   {repr(chunk.page_content[:50])}...")

    # 4.生成摘要
    # 创建摘要生成链
    chain = (
            {"chunk": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("总结下面的文档:\n\n{chunk}")
            | llm
            | StrOutputParser()
    )

    # 批量生成文档摘要（最大并发数5）
    summaries = chain.batch(chunks, {"max_concurrency": 5})

    for i, summary in enumerate(summaries):
        print(f"块 {i + 1} :   {repr(summary[:50])}...")

    # 5.索引准备
    #   InMemoryByteStore 是一个内存中的存储层，用于存储原始文档
    #   Chroma 是一个文档向量数据库，用于存储文档摘要的向量表示
    # 初始化Chroma实例（用于存储摘要向量）
    vectorstore = Chroma(
        collection_name="summaries",
        embedding_function=embeddings_model
    )

    # 初始化内存字节存储（用于存储原始文档）
    store = InMemoryByteStore()

    # 6.索引构建
    # 初始化多向量检索器（结合向量存储和文档存储）
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": 1}
    )

    # 为每个文档生成唯一ID
    doc_ids = [str(uuid.uuid4()) for _ in chunks]

    # 创建摘要文档列表（包含生成的唯一ID作为对应摘要文档的元数据）
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    # 将摘要添加到向量数据库
    retriever.vectorstore.add_documents(summary_docs)

    # 将原始文档存储到字节存储（使用ID关联）
    retriever.docstore.mset(list(zip(doc_ids, chunks)))

    # 7.检索
    def pretty_print_docs(docs):
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )

    contexts = retriever.invoke('deepseek的企业动态')
    pretty_print_docs(contexts)


test_no_summary()
test_summary()
test_summary_index()
