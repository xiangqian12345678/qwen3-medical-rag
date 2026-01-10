from sentence_transformers.cross_encoder import CrossEncoder
import os

"""
Cross-Encoder Reranker 工作原理
1. 基本概念
    Cross-Encoder（交叉编码器）是一种能够同时处理两个文本输入并直接输出相关性分数的神经网络模型。
    与Bi-Encoder（双编码器）不同，Cross-Encoder能够捕捉查询和文档之间的细粒度交互。
2. 架构细节
    典型的Cross-Encoder基于Transformer架构：
    输入层：查询和文档被拼接成一个序列，格式通常为：[CLS] query [SEP] document [SEP]
    Transformer编码器：多个Transformer层处理输入序列
    输出层：取[CLS]标记的表示，通过一个线性层输出分数
3. 训练过程
    Cross-Encoder通常通过监督学习进行训练：
    训练数据：包含(查询, 正例文档, 负例文档)的三元组
    损失函数：常用对比损失(如margin loss)或排序损失(如pairwise hinge loss)
    目标：使正例文档的分数高于负例文档
4. 推理过程 
    查询和每个候选文档组成输入对
    模型计算每个对的相关性分数
    根据分数对所有候选文档进行重新排序
"""

# 1.CrossEncoder模型初始化
MODEL_DIR = "/mnt/c/大模型/智泊大模型全栈教程总结/02-教材整理 L2/代码/Langchain/data/BAAI"
model_name = "bge-reranker-large"
model_path = os.path.join(MODEL_DIR, model_name)
reranker = CrossEncoder(model_path, device="cpu")

# 2.相关性计算
query = "孕妇感冒了怎么办"
documents = [
    "感冒应该吃999感冒灵",
    "高血压患者感冒了吃什么",
    "感冒了可以吃感康，但是孕妇禁用"
]

pairs = [[query, doc] for doc in documents]
# 模型为每个查询-文档对计算一个相关性分数。
scores = reranker.predict(pairs)
print(scores)
