import os
import uuid
from typing import List

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryByteStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy import dot
from numpy.linalg import norm
from pydantic.v1 import BaseModel, Field


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

# 2.模型准备 模型能力很关键
llm = ChatOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com/v1",
                 model="deepseek-chat")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

# 3.加载数据
loader = TextLoader(TXT_DOCUMENT_PATH, encoding='utf-8')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = text_splitter.split_documents(docs)


# 4.生成假设性问题准备
class HypotheticalQuestions(BaseModel):
    """生成假设性问题"""
    questions: List[str] = Field(..., description="List of questions")


prompt = ChatPromptTemplate.from_template(
    """生成一个包含3个假设问题的列表，以下文档可用于回答这些问题:

    {doc}
    """
)

chain = (
        {"doc": lambda x: x.page_content}
        | prompt
        | llm.with_structured_output(HypotheticalQuestions) # Tool 模式对 JSON 严格要求 → 中文 + 长文本必炸
        | (lambda x: x.questions)
)

# 5. 构建假设性问题索引
# 批量处理所有文档生成假设性问题（最大并行数5）
hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})
print(hypothetical_questions)

# 初始化Chroma向量数据库（存储生成的问题向量）
vectorstore = Chroma(
    collection_name="hypo-questions", embedding_function=embeddings_model
)

# 初始化内存存储（存储原始文档）
store = InMemoryByteStore()
# 文档标识键名
id_key = "doc_id"
# 配置多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
    search_kwargs={"k": 1}
)

# 为每个原始文档生成唯一ID
doc_ids = [str(uuid.uuid4()) for _ in docs]
# 将生成的问题转换为带元数据的文档对象
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend([Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list])

retriever.vectorstore.add_documents(question_docs)  # 将问题文档存入向量数据库
retriever.docstore.mset(list(zip(doc_ids, docs)))  # 将原始文档存入字节存储（通过ID关联）

# 6. 检索
sub_docs = retriever.vectorstore.similarity_search("deepseek受到哪些攻击？")
print("问题：deepseek受到哪些攻击？")
print("similarity_search检索结果：")
print(sub_docs[0].page_content)

retrieved_docs = retriever.invoke("deepseek受到哪些攻击？")
print("问题：deepseek受到哪些攻击？")
print("invoke检索结果：")
print(retrieved_docs[0].page_content)

# 7.演示案例
docs = [
    """团队协作中的常见障碍：
        沟通不畅：团队成员之间缺乏有效的沟通，导致信息传递不准确或不及时。
        目标不明确：团队成员对共同目标的理解不一致，导致工作方向不一致。
        缺乏信任：团队成员之间缺乏信任，导致合作效率低下。
        资源分配不均：团队资源分配不合理，导致部分成员工作量过大，部分成员闲置。""",

    """提高团队协作效率的策略：
        明确目标与分工：确保每个团队成员都清楚团队的共同目标和自己的具体任务。
        建立有效的沟通机制：定期召开团队会议，使用协作工具（如Slack、Microsoft Teams）保持实时沟通。
        培养团队信任：通过团队建设活动和透明的沟通机制，增强团队成员之间的信任。
        合理分配资源：根据团队成员的技能和工作量，合理分配任务和资源。""",

    """远程团队协作的最佳实践：
        使用协作工具：利用Zoom、Google Workspace等工具进行远程会议和文档协作。
        定期检查进度：通过定期的进度报告和检查，确保远程团队的工作进展顺利。
        建立明确的沟通规范：制定远程团队的沟通规则，确保信息传递的准确性和及时性。"""
]

query = "团队沟通不畅怎么办"
doc_embeddings = embeddings_model.embed_documents(docs)
query_embedding = embeddings_model.embed_query(query)
similarities = [dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))
                for doc_embedding in doc_embeddings]

# 8.未来使用假设性问题
print("==========================未使用假设性问题索引==========================")
for i, similarity in enumerate(similarities):
    print(f"第{i + 1}个相似度：", similarity)

# 9.使用假设性问题索引
chain = (
        {"doc": lambda x: x}
        | prompt
        | llm.with_structured_output(HypotheticalQuestions)
        | (lambda x: x.questions)
)

hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})
for q in hypothetical_questions:
    print(q)
    print("=" * 150)

print("==========================使用假设性问题索引==========================")
for hypothetical_question in hypothetical_questions:
    hypothetical_question_embeddings = embeddings_model.embed_documents(
        [question for question in hypothetical_question])
    similarities = [
        dot(query_embedding, hypothetical_embedding) / (norm(query_embedding) * norm(hypothetical_embedding)) for
        hypothetical_embedding in hypothetical_question_embeddings]
    print(similarities)
