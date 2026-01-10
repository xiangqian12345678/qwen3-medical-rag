from langchain_core.documents import Document

# BM25检索结果：
bm25_documents = [
    Document(page_content="人工智能在不同行业的应用", id="1"),
    Document(page_content="这款应用在智能手机上非常流行，用户可以用来购物、支付和社交。", id="2"),
    Document(page_content="老师将新的教学方法应用到课堂中，学生的学习积极性明显提高了。", id="3"),
]

# 向量检索结果：
vector_documents = [
    Document(page_content="人工智能在不同行业的应用", id="1"),
    Document(page_content="老师将新的教学方法应用到课堂中，学生的学习积极性明显提高了。", id="3"),
    Document(page_content="公司正在开发一款新的应用，旨在提高员工的工作效率。", id="4"),
]

documents = [bm25_documents, vector_documents]


def rrf_rank(documents: list[list[Document]], k=60) -> list[Document]:
    # 初始化rrf字典（key=文档id，value={"rrf_score":累计分数,"doc":文档对象}）
    # k: 平滑常数，通常设置为60
    rrf_scores = {}
    # 遍历每个检索结果列表（每个查询对应的结果）
    for docs in documents:
        # 为每个文档列表计算排名（从1开始）
        for rank, doc in enumerate(docs, 1):
            # 计算当前文档的RRF分数
            rrf_score = 1 / (k + rank)
            # 如果文档已经在字典中，累加RRF分数
            if doc.id in rrf_scores:
                rrf_scores[doc.id]['rrf_score'] += rrf_score
            else:
                rrf_scores[doc.id] = {'rrf_score': rrf_score, 'doc': doc}

    # 将字典转换为列表，并根据字段value：RRF分数排序
    sorted_docs = sorted(
        rrf_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True  # 降序排列：从大到小
    )
    for item in sorted_docs:
        print(item)
    result = [item['doc'] for item in sorted_docs]
    return result


rrf_rank(documents)
