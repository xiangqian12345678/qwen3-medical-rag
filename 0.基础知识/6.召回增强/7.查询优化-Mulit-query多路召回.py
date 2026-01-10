# 相当于一次查询 → 多次检索 → 合并结果 → 去重
# 它让 LLM 把原始 query 改写为一组语义不同但相关的问题：
#   比如 LLM 生成：
#       deepseek 遭遇了哪些质疑？
#       哪些国家限制或封禁了 deepseek？
#       deepseek 在国际上面临哪些挑战？
#       deepseek 是否遇到安全类指控？
#   然后 用这4个问题全部去向量检索
import logging
import os

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 日志句柄初始化
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# 1.文件路径
RESOURCE_DIR = "/mnt/c/大模型/智泊大模型全栈教程总结/02-教材整理 L2/代码/Langchain/6.langchain高级RAG/data/resources"
TXT_DOCUMENT_PATH = os.path.join(RESOURCE_DIR, "deepseek百度百科.txt")

# 2.模型初始化
llm = ChatTongyi(model="qwen-max")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

# 3.加载文档
loader = TextLoader(TXT_DOCUMENT_PATH, encoding='utf-8')
docs = loader.load()

# 4.召回引擎创建与数据索引
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=chunks,
                                    embedding=embeddings_model,
                                    collection_name="multi-query")
retriever = vectorstore.as_retriever()

# 5. 多路召回
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
        You are an AI language model assistant. 
        Your task is to generate 5 different versions of the given user 
        question to retrieve relevant documents from a vector  database. 
        By generating multiple perspectives on the user question, 
        your goal is to help the user overcome some of the limitations 
        of distance-based similarity search. 
        Provide these alternative questions separated by newlines. 
        Original question: {question}
    """
)

retrieval_from_llm = MultiQueryRetriever.from_llm(
    prompt=QUERY_PROMPT,
    retriever=retriever,
    llm=llm,
    include_original=True  # 是否包含原始查询
)

# 自动去重
unique_docs = retrieval_from_llm.invoke("详细介绍DeepSeek")
pretty_print_docs(unique_docs)

# 6. 答案合成
# 创建prompt模板
template = """
请根据以下文档回答问题:
### 文档:
{context}
### 问题:
{question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm
response = chain.invoke({"context": [doc.page_content for doc in unique_docs], "question": "详细介绍DeepSeek"})
print('-'*20 + "答案合成" + '-'*20)
print(response.content)
