import base64
import os
import re
import shutil
import uuid

from IPython.display import HTML, display, Markdown
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from unstructured.documents.elements import Image as ImageElement
from unstructured.partition.pdf import partition_pdf

# 环境配置
os.environ["TESSERACT_CMD"] = "/usr/bin/tesseract"

# 1. 初始化
BASE_DIR = "/mnt/c/大模型/智泊大模型全栈教程总结/02-教材整理 L2/代码/Langchain/6.langchain高级RAG/data/"
RESOURCE_DIR = BASE_DIR + 'resources/'
TOOLS_DIR = BASE_DIR + 'tools/'
IMAGE_OUT_DIR = RESOURCE_DIR + 'images/'
PDF_PATH = RESOURCE_DIR + "978-7-5170-2271-8_1.pdf"
RESIZE_IMAGE_DIR = RESOURCE_DIR + "temp"

# 2. 加载pdf
# 如果图片提取目录存在则删除重建
if os.path.exists(IMAGE_OUT_DIR):
    shutil.rmtree(IMAGE_OUT_DIR)
os.makedirs(IMAGE_OUT_DIR)

# 使用unstructured库解析PDF文档(需要科学上网)
pdf_data = partition_pdf(
    filename=PDF_PATH,
    extract_images_in_pdf=True,
    infer_table_structure=True,  # 启用表格结构识别
    max_characters=4000,  # 每个文本块最大字符数
    new_after_n_chars=3800,  # 达到3800字符后分新块
    combine_text_under_n_chars=2000,  # 合并小于2000字符的文本块
    chunking_strategy="by_title",  # 按标题分块
    extract_image_block_output_dir=IMAGE_OUT_DIR,  # 图片提取路径
)

# 3. 生成摘要  生成文本和表格摘要
prompt = PromptTemplate.from_template(
    "你是一位负责生成表格和文本摘要以供检索的助理。"  # 角色
    "这些摘要将被嵌入并用于检索原始文本或表格元素。"
    "请提供表格或文本的简明摘要，该摘要已针对检索进行了优化。表格或文本：{document}"  # 任务
)

# 使用大模型生成文本摘要
model = ChatTongyi(model="qwen-max")
summarize_chain = {"document": lambda x: x.text} | prompt | model | StrOutputParser()
summaries = summarize_chain.batch(pdf_data, {"max_concurrency": 5})


def image_summarize(image_path):
    """生成图片摘要"""
    chat = ChatTongyi(model="qwen-vl-max")
    local_image_path = f"file://{image_path}"
    print(f"生成摘要:{image_path}")
    response = chat.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "text": "你是一名负责生成图像摘要以便检索的助理。这些摘要将被嵌入并用于检索原始图像。请生成针对检索进行了优化的简洁的图像摘要。"},
                    {"image": local_image_path}
                ]
            )
        ]
    )
    return response.content


image_summaries = []
# image_list = []
for element in pdf_data:
    orig_elements = element.metadata.orig_elements
    length = len(orig_elements)
    for i, orig_element in enumerate(orig_elements):
        # 图片元素
        if isinstance(orig_element, ImageElement):
            image_path = orig_element.metadata.to_dict()["image_path"]
            # image_list.append(orig_element)
            # 将图片摘要记录在图片元素的text属性中
            summarizes = image_summarize(image_path)[0]["text"]
            orig_element.text = summarizes
            image_summaries.append(summarizes)

# 4. 构建索引
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")
# 创建向量数据库（用于存储摘要）
vectorstore = Chroma(
    collection_name="multi_model_opt",
    embedding_function=embeddings_model
)
# 创建内存存储（用于存储原内容）
docstore = InMemoryStore()
# 将摘要存储入库
id_key = "doc_id"


def add_documents(doc_summaries, doc_contents):
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    vectorstore.add_documents(summary_docs)
    docstore.mset(list(zip(doc_ids, doc_contents)))


# 不再单独放入图片摘要??
add_documents(summaries, pdf_data)  # PDF Element

# 5. 构建多向量检索（摘要索引）
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
    search_kwargs={"k": 2}
)


# 6. RAG构建
def split_image_text_types(docs):
    texts = []
    # 注意：doc为PDF Element
    for doc in docs:
        text = ''
        # 从文档元数据中获取原始元素列表，这些元素可能包含文本、图片等不同类型
        orig_element = doc.metadata.orig_elements
        for element in orig_element:
            # 是否为图片元素
            if isinstance(element, ImageElement):
                # 图片元素用标记包裹  src=图片路径  element.text=图片相关文本  例如： <image src="/path/to/image.png">图片描述文字</image>
                text += f'<image src="{element.metadata.image_path}">{element.text}</image>'
            else:
                # 其他元素直接放入文本
                text += element.text
        texts.append(text)
    return texts


def prompt_func(data_dict):
    question = data_dict["question"]
    context = data_dict["context"]
    # 装载数据
    formatted_texts = "\n\n".join(context)

    prompt = ("你是作为一名专业的电气工程师和电路理论专家。你的任务是用中文回答与电路基本概念和定律相关的问题。"
              "你将获得相关文档作为参考的上下文。这些文档都是根据用户输入的关键词从向量数据库中检索获取的。"
              "请根据提供的文档结合你丰富的知识与分析能力，提供一份全面的问题解答。"
              r"请返回Markdown格式数据，并且当涉及到数学公式时，请使用正确的LaTeX语法来编写这些公式，对于行内公式应该以单个美元符号`$`;对于独立成行的公式，使用双美元符号`$$`包裹。例如，行内公式：`$a = \frac{1}{2}$`,而独立成行的公式则是：`$$ a = \frac{1}{2} $$`。"
              "请将提供的文档中的图片`<image src='...'>`(不包括`<image>`)到`</image>`中间的文字，在需要时自然融入答案阐述的对应位置进行合理排版。\n\n"
              f"用户的问题是：\n{question}\n\n"
              "参考的文本或者表格数据：\n"
              f"{formatted_texts}")

    print('*' * 200)
    print(f"prompt: {prompt}")
    print('*' * 200)

    return prompt


llm = ChatTongyi(model="qwen-max")
chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(split_image_text_types)
        }
        | RunnableLambda(prompt_func)
        | llm
        | StrOutputParser()
)

# 7. 效果展示
result = chain.invoke("介绍下电路模型")


def encode_image(img_path):
    with open(img_path, "rb") as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return img_base64


def display_answer(text: str):
    # 正则表达式：用于匹配 <image src="xxx"> 并捕获 src 属性的值
    pattern = r'(<image src="([^"]*)">)'
    # abc<image src="xxx">def -》 ["abc", "<image src="xxx">", "xxx", "def"]
    chunks = re.split(pattern, text)
    for i, chunk in enumerate(chunks):
        # 文本内容
        if i % 3 == 0:
            display(Markdown(chunk.replace("</image>", "")))
        elif i % 3 == 2:
            # image_path = re.search(r'src="([^"]*)"', chunk).group(1) 1% 3==1
            img_base64 = encode_image(chunk)
            display(HTML(f'\n<img src="data:image/jpeg;base64,{img_base64}"/>\n'))


print('RAG结果展示:' + '-' * 100)
display_answer(result)
