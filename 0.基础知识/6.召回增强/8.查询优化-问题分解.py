from typing import List

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import LineListOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate, BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda


# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# 1. 模型初始化
llm = ChatTongyi(model="qwen-max")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

# 2. 数据准备
documents = [
    Document(
        page_content="""番茄炒蛋的食材：\n
        - 新鲜鸡蛋：3-4个（根据人数调整）
        - 番茄：2-3个中等大小\n- 盐：适量
        - 白糖：一小勺（可选，用于提鲜）
        - 食用油：适量
        - 葱花：少许（可选，用于增香）\n
        这些是最基本的材料，当然也可以根据个人口味添加其他调料或配料。
        """),
    Document(
        page_content="""番茄炒蛋的步骤：鸡蛋打入碗中，加入少许盐，用筷子或打蛋器充分搅拌均匀；
           - 番茄洗净后切成小块备用。\n
           3. **炒鸡蛋**：锅内倒入适量食用油加热至温热状态，然后将搅拌好的鸡蛋液缓缓倒入锅中。
           待鸡蛋凝固时轻轻翻动几下，让其受热均匀直至完全熟透，随后盛出备用。\n
           4. **炒番茄**：在同一锅里留下的底油中放入切好的番茄块，中小火慢慢翻炒至出汁，可根据个人口味加一点点白糖提鲜。\n
           5. **合炒**：当番茄炒至软烂并开始释放大量汤汁时，再把之前炒好的鸡蛋倒回锅里，快速与番茄混合均匀，同时加入适量的盐调味。
           如果喜欢的话还可以撒上一些葱花增加香气。\n
           6. **完成**：最后检查一下味道是否合适，确认无误后即可关火装盘享用美味的番茄炒蛋啦！
           """),
    Document(
        page_content="""技巧与注意事项：
        1. **选材**：选择新鲜的鸡蛋和成熟的番茄。新鲜的食材是做好这道菜的基础。
        2. **打蛋液**：将鸡蛋打入碗中后加入少许盐（根据个人口味调整），然后充分搅拌均匀。这样做可以让蛋更加松软且味道更佳。
        3. **处理番茄**：番茄最好先用开水稍微焯一下皮，然后去皮切块。这样可以去除表皮的硬质部分，让番茄更容易入味，并且口感更好。
        4. **热锅冷油**：先用中小火把锅烧热，再倒入适量食用油，待油温五成热时下蛋液。这样的做法可以使蛋快速凝固形成漂亮的形状而不易粘锅。
        5. **分步烹饪**：通常建议先炒鸡蛋至半熟状态取出备用；接着利用剩下的底油继续翻炒番茄至出汁，
        最后再将之前炒好的鸡蛋倒回锅里与番茄混合均匀加热即可。
        6. **调味品**：除了基本的盐之外，还可以根据喜好添加少量糖来提鲜或者一点酱油增色添香。注意调味料不宜过多以免掩盖了食材本身的味道。
        7. **出锅前加葱花**：如果喜欢的话，在即将完成时撒上一些葱花不仅能增加菜品色泽还能增添香气。
        """)
]

# 3. 数据向量化存储
vectorstore = Chroma.from_documents(documents=documents,
                                    embedding=embeddings_model,
                                    collection_name="decomposition")

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
pretty_print_docs(retriever.invoke("番茄炒蛋怎么制作？"))

#  4. 问题分解
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to break down the input question into 3 sub-questions,
     and solve the complete problem by solving these sub-questions one by one.
     The sub-questions need to retrieve relevant documents in the vector database.
     By decomposing the user's question to generate sub-questions, 
     your goal is to help users overcome some limitations of distance-based similarity search.
     Provide these sub-questions separated by newlines, no additional content is required. Original question: {question}""",
)

chain = QUERY_PROMPT | llm | LineListOutputParser()
questions = chain.invoke({"question": "番茄炒蛋怎么制作？"})
print('-' * 20 + '分解问题' + '-' * 20)
print(questions)

# 5. 问题分解整合功能类构建
SUB_QUESTION_PROMPT = PromptTemplate(
    input_variables=["question", "sub_question", "documents"],
    template="""
    To address the main problem {question}, 
    you need to first resolve the sub-question {sub_question}. 
    Below is the reference document provided to support your reasoning:\n\n{documents}\n\n 
    Please provide the answer to the current sub-question directly.""",
)

class DecompositionQueryRetriever(BaseRetriever):
    # 向量数据库检索器
    retriever: BaseRetriever
    # 生成子问题链
    llm_chain: Runnable
    # 解决子问题链
    sub_llm_chain: Runnable

    @classmethod
    def from_llm(
            cls,
            retriever: BaseRetriever,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = QUERY_PROMPT,
            sub_prompt: BasePromptTemplate = SUB_QUESTION_PROMPT
    ) -> "DecompositionQueryRetriever":
        output_parser = LineListOutputParser()
        llm_chain = prompt | llm | output_parser
        sub_llm_chain = sub_prompt | llm
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            sub_llm_chain=sub_llm_chain
        )

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 生成子问题
        sub_queries = self.generate_queries(query)
        # 解决子问题
        documents = self.retrieve_documents(query, sub_queries)
        return documents

    def generate_queries(self, question: str) -> List[str]:
        response = self.llm_chain.invoke({"question": question})
        lines = response
        print(f"Generated queries: {lines}")
        return lines

    def retrieve_documents(self, query: str, sub_queries: List[str]) -> List[Document]:
        sub_llm_chain = RunnableLambda(
            # 传入子问题，检索文档并回答
            lambda sub_query: self.sub_llm_chain.invoke(
                {
                    "question": query,
                    "sub_question": sub_query,
                    "documents": [doc.page_content for doc in self.retriever.invoke(sub_query)]
                }
            )
        )
        # 批量执行所有的子问题
        responses = sub_llm_chain.batch(sub_queries)
        # 将子问题和答案合并作为解决主问题的文档
        documents = [
            Document(page_content=sub_query + "\n" + response.content)
            for sub_query, response in zip(sub_queries, responses)
        ]
        return documents


# 6. LangChain问题分解整合测试
decompositionQueryRetriever = DecompositionQueryRetriever.from_llm(llm=llm, retriever=retriever)
decomposition_docs = decompositionQueryRetriever.invoke("番茄炒蛋怎么制作？")
pretty_print_docs(decomposition_docs)

# 8. 根据召回的文档解答问题
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

response = chain.invoke({"context": [doc.page_content for doc in decomposition_docs], "question": "番茄炒蛋怎么制作？"})
print('-' * 20 + '大模型回答' + '-' * 20)
print(response.content)
