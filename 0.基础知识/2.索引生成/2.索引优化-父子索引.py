import os

from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# 1.文件路径
RESOURCE_DIR = "../../data/base/resources"
TXT_DOCUMENT_PATH = os.path.join(RESOURCE_DIR, "deepseek百度百科.txt")

# 2.初始化模型
llm = ChatTongyi(model="qwen-max")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

# 3.加载本地数据
loader = TextLoader(TXT_DOCUMENT_PATH, encoding='utf-8')
docs = loader.load()
# 创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_overlap=32, chunk_size=256)
# 创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_overlap=16, chunk_size=64)

# 4.存储准备
# 存储小块
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=embeddings_model
)
# 创建内存存储对象，存储大块
store = InMemoryStore()

# ================================4. 创建检索器================================
# 创建父文档检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 1}
)

# 5.存储文档
retriever.add_documents(docs)

# 6.子索引检索
sub_docs = vectorstore.similarity_search("介绍下DeepSeek和市场占用情况")
print("-" * 20 + "子索引检索" + "-" * 20)
for sub_doc in sub_docs:
    print(sub_doc.page_content)

# 7.父索引检索
print("-" * 20 + "父索引检索" + "-" * 20)
retrieved_docs = retriever.invoke("介绍下DeepSeek和市场占用情况")
for retrieved_doc in retrieved_docs:
    print(retrieved_doc.page_content)
