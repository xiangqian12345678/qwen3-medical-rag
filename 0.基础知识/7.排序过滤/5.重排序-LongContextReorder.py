from langchain_community.document_transformers import LongContextReorder

"""
根据论文（Lost in the Middle: How Language Models Use Long Contexts）当关键数据位于输入上下文的开头或结尾时，
通常会获得最佳性能。为了减轻 “lost in the middle”的影响，可以在检索后重新排序文档，使最相关的文档置于极值
（例如，上下文的第一和最后一部分），将最不相关的文档置于中间。
"""
# 5,4,3,2,1
# 倒排：1,2,3,4,5
# index%2=0: 往第一个放，index%2=1 往最后放

documents = [
    "相关性:5",
    "相关性:4",
    "相关性:3",
    "相关性:2",
    "相关性:1",
]

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(documents)

print(reordered_docs)
