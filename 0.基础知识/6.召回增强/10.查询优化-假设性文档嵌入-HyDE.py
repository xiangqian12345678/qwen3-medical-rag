import os

from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
核心思想
HyDE（Hypothetical Document Embeddings 假设性文档嵌入） 是一种无需真实文档的检索增强生成（RAG）技术。
其核心是通过模型先生成假设性答案，再根据该假设答案的嵌入向量去检索真实文档。

关键步骤
    1. 生成假设答案：
        让模型基于问题生成一个假设的答案（无需准确，只需语义相关）。
        示例：
        问题 → "如何训练一只猫用马桶？"
        假设答案 → "训练猫用马桶需要逐步引导，首先将猫砂盆靠近马桶..."
    2. 嵌入假设答案：
        将假设答案转换为向量（如用 OpenAI embeddings）。
    3. 向量检索：
        用该向量在数据库中检索真实相关的文档。
    4. 生成最终答案：
        结合检索到的真实文档生成可靠回答。
"""

# 1.文件路径
RESOURCE_DIR = "/mnt/c/大模型/智泊大模型全栈教程总结/02-教材整理 L2/代码/Langchain/6.langchain高级RAG/data/resources"
TXT_DOCUMENT_PATH = os.path.join(RESOURCE_DIR, "deepseek百度百科.txt")

# 2.模型初始化
llm = ChatTongyi(model="qwen-max")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

# 3.加载文档
loader = TextLoader(TXT_DOCUMENT_PATH, encoding='utf-8')
docs = loader.load()

# 4.文档向量化并存储
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=chunks,
                                    embedding=embeddings_model,
                                    collection_name="multi-query")
retriever = vectorstore.as_retriever()

# 4.生成假设答案
hyde_prompt = """
    请根据问题生成一个假设性答案（无需准确）：
    问题: {question}
    假设答案:
    """
question = "详细介绍DeepSeek"
hypothetical_answer = llm.invoke(hyde_prompt.format(question=question))
print(f"假设答案：\n {hypothetical_answer.content}")

# 5. 基于答案检索文档
retrieved_docs = vectorstore.similarity_search(hypothetical_answer.content, k=3)
print(f"召回文档：\n {retrieved_docs}")

# 6. 基于召回文档生成答案
final_prompt = f"""
    基于以下真实文档回答问题：
    文档: {retrieved_docs}
    问题: {question}
    答案:
    """
print(f"最终提示：\n{final_prompt}")
result = llm.invoke(final_prompt)
print('-' * 20 + '基于召回文档生成答案' + '-' * 20)
print(result)
