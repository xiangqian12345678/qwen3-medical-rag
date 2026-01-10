import os

from langchain_chroma import Chroma
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


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

# 2.模型准备 模型能力很关键
llm = ChatOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com/v1",
                 model="deepseek-chat")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

# 3.数据准备==========================
docs = [
    Document(
        page_content="小米智能手环6",
        metadata={"品牌": "小米", "价格": 249, "评分": 4.6}
    ),
    Document(
        page_content="华为FreeBuds Pro无线耳机",
        metadata={"品牌": "华为", "价格": 999, "评分": 4.8}
    ),
    Document(
        page_content="小米移动电源3",
        metadata={"品牌": "小米", "价格": 99, "评分": 4.4}
    ),
    Document(
        page_content="华为Mate 40 Pro智能手机",
        metadata={"品牌": "华为", "价格": 6999, "评分": 5.0}
    ),
    Document(
        page_content="小米AirDots Pro蓝牙耳机",
        metadata={"品牌": "小米", "价格": 299, "评分": 4.5}
    ),
    Document(
        page_content="华为智能手表GT 2",
        metadata={"品牌": "华为", "价格": 1288, "评分": 4.7}
    ),
    Document(
        page_content="小米小爱音箱Play",
        metadata={"品牌": "小米", "价格": 169, "评分": 4.3}
    )
]

# 4. 构建元数据索引
metadata_field_info = [
    {"name": "品牌", "type": "string", "description": "产品的品牌名称"},
    {"name": "价格", "type": "integer", "description": "产品的价格"},
    {"name": "评分", "type": "float", "description": "产品的用户评分"},
]

# 文档内容描述（指导LLM理解文档内容）
document_content_description = "电子产品的信息"
vectorstore = Chroma.from_documents(docs, embeddings_model, collection_name="self-query")

# 5.创建自查询检索器（核心组件）
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

# 6.检索
pretty_print_docs(retriever.invoke("华为价格5000以上的商品"))
