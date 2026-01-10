import os

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import DashScopeEmbeddings

"""
1. CrossEncoderReranker 初始化
    compressor = CrossEncoderReranker(model=model, top_n=3)
    作用：创建一个基于交叉编码器(Cross-Encoder)的重排序器
    参数解析：
        model: 使用的重排序模型实例(这里是HuggingFaceCrossEncoder)
        top_n: 指定保留前多少个重排序后的结果(这里设为3)

    工作原理：
        接收初始检索结果(通常来自向量检索)
        对每个文档与查询(query)的相关性进行精细评分
        根据评分重新排序文档
        只保留top_n个最相关的文档

2. ContextualCompressionRetriever 创建
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    作用：将基础检索器和重排序器组合成一个压缩检索器
    参数解析：
        base_compressor: 上面创建的CrossEncoderReranker实例
        base_retriever: 基础检索器(这里是Chroma向量检索器)

工作流程：
    首先使用base_retriever获取初步检索结果
    然后使用base_compressor对这些结果进行重排序和过滤

3. 检索过程
    compressed_docs = compression_retriever.invoke("人工智能的应用")
    执行流程：
        向量检索器(retriever)首先找到与"人工智能的应用"相关的文档(基于向量相似度)
        重排序器(compressor)对这些文档进行更精细的相关性评估：
        计算查询与每个文档的交叉注意力
        生成更准确的相关性分数
        根据重排序分数，只保留前3个最相关的文档
"""


# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# 1. 数据准备
texts = [
    "人工智能在医疗诊断中的应用。",
    "人工智能如何提升供应链效率。",
    "NBA季后赛最新赛况分析。",
    "传统法式烘焙的五大技巧。",
    "红楼梦人物关系图谱分析。",
    "人工智能在金融风险管理中的应用。",
    "人工智能如何影响未来就业市场。",
    "人工智能在制造业的应用。",
    "今天天气怎么样",
    "人工智能伦理：公平性与透明度。"
]

# 2.文档向量化并存储
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings_model,
    collection_name="rrf"
)
retriever = vectorstore.as_retriever()

# 3.排序模型
MODEL_DIR = "/mnt/c/大模型/智泊大模型全栈教程总结/02-教材整理 L2/代码/Langchain/data/BAAI"
model_name = "bge-reranker-large"
model_path = os.path.join(MODEL_DIR, model_name)
model = HuggingFaceCrossEncoder(model_name=model_path, model_kwargs={'device': 'cpu'})

# 4.重排序模型初始化
compressor = CrossEncoderReranker(model=model, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever  # retriever = 混合检索 或 multi-query
)

# 5.召回并重排序
compressed_docs = compression_retriever.invoke("人工智能的应用")
pretty_print_docs(compressed_docs)
