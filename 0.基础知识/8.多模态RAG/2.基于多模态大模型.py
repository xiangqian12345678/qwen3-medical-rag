import base64
import os
import shutil
import uuid
from pathlib import Path

import markdown
from IPython.display import HTML, display
from PIL import Image
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
from unstructured.documents.elements import Table, CompositeElement
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


def is_image_path(filepath):
    ''' 判断是否为图片地址 '''
    try:
        path = Path(filepath)
        return all([
            path.exists(),
            path.is_file(),
            path.suffix.lower() == '.jpg'
        ])
    except Exception:
        return False


def resize_base64_image4tongyi(image_path, max_size=(640, 480)):
    try:
        # 打开图片
        img = Image.open(image_path)
        width, height = img.size
        ratio = min(max_size[0] / width, max_size[1] / height)
        # 计算按比例缩放后的宽高
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # 缩放
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        # 保存图片到指定路径
        out_path = out_path = os.path.join(RESIZE_IMAGE_DIR, str(uuid.uuid4()) + ".jpg")
        resized_img.save(out_path)
        # 图片地址
        return out_path
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None


def split_image_text_types(docs):
    """
    拆分图像和文本
    """
    images = []
    texts = []
    # docs：文本内容和图片地址
    for doc in docs:
        if is_image_path(doc):
            doc = resize_base64_image4tongyi(doc)
            images.append(doc)
        else:
            texts.append(doc)
    return {"images": images, "texts": texts}


def show_plt_img(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))


def encode_image(img_path):
    with open(img_path, "rb") as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return img_base64


def display_answer(text: str):
    start_tag = "<image>"
    end_tag = "</image>"
    parts = text.split(start_tag)
    for part in parts:
        # 再根据</image>标签 分割文本：xxx => ['xxx'], xxx</image> => ['xxx','']
        chunks = part.split(end_tag)
        if len(chunks) > 1:
            # 存在图片
            image_path = chunks[0]
            context = chunks[1]
            img_base64 = encode_image(image_path)
            display(HTML(f'\n<img src="data:image/jpeg;base64,{img_base64}"/>\n'))
            # display(HTML(context.replace("\n", "<br/>")))
            display(HTML(markdown.markdown(context)))
        else:
            display(HTML(markdown.markdown(part)))


# 2. 数据准备
# 如果图片提取目录存在则删除重建
if os.path.exists(IMAGE_OUT_DIR):
    shutil.rmtree(IMAGE_OUT_DIR)
os.makedirs(IMAGE_OUT_DIR)

print('\n开始解析pdf文档' + '-' * 100)
pdf_data = partition_pdf(
    filename=PDF_PATH,
    extract_images_in_pdf=True,  # 提取图片
    infer_table_structure=True,  # 启用表格结构识别
    max_characters=4000,  # 每个文本块最大字符数
    new_after_n_chars=3800,  # 达到3800字符后分新块
    combine_text_under_n_chars=2000,  # 合并小于2000字符的文本块
    chunking_strategy="by_title",  # 按标题分块
    extract_image_block_output_dir=IMAGE_OUT_DIR,  # 图片提取路径
)


# 3. 提取表格与文本
tables = []
texts = []
for element in pdf_data:
    if isinstance(element, Table):
        tables.append(str(element))
    elif isinstance(element, CompositeElement):
        texts.append(str(element))
print(f"表格元素：{len(tables)}  文本元素：{len(texts)}")

# 4. 生成文本和表格摘要
prompt = PromptTemplate.from_template(
    "你是一位负责生成表格和文本摘要以供检索的助理。"  # 角色
    "这些摘要将被嵌入并用于检索原始文本或表格元素。"
    "请提供表格或文本的简明摘要，该摘要已针对检索进行了优化。表格或文本：{document}"  # 任务
)

# 使用大模型生成文本摘要
model = ChatTongyi(model="qwen-max")
summarize_chain = {"document": lambda x: x} | prompt | model | StrOutputParser()
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})


# 5. 生成图片摘要
def image_summarize(image_path):
    """生成图片摘要"""
    chat = ChatTongyi(model="qwen-vl-max")
    local_image_path = f"file://{image_path}"
    response = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"text": "你是一名负责生成图像摘要以便检索的助理。这些摘要将被嵌入并用于检索原始图像。"  # 角色
                             "请生成针对检索进行了优化的简洁的图像摘要。"},  # 任务
                    {"image": local_image_path}
                ]
            )
        ]
    )
    return response.content


# 检索图片摘要获得图片地址
img_list = []  # 原图片地址
image_summaries = []  # 图片摘要
for img_file in sorted(os.listdir(IMAGE_OUT_DIR)):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(IMAGE_OUT_DIR, img_file)
        img_list.append(img_path)
        # 生成图片摘要
        image_summaries.append(image_summarize(img_path)[0]["text"])

# 6. 构建向量索引（摘要索引）
embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v1",
)

# 创建向量数据库（用于存储摘要）
vectorstore = Chroma(
    collection_name="multi_model",
    embedding_function=embeddings_model
)

# 创建内存存储（用于存储原内容）
docstore = InMemoryStore()
# 将摘要存储入库
id_key = "doc_id"


def add_documents(doc_summaries, doc_contents):
    if not doc_summaries or not doc_contents:
        print("警告：文档摘要或内容为空，跳过添加操作。")
        return

    if len(doc_summaries) != len(doc_contents):
        print("警告：文档摘要和内容的数量不匹配，跳过添加操作。")
        return

    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    vectorstore.add_documents(summary_docs)
    docstore.mset(list(zip(doc_ids, doc_contents)))


add_documents(text_summaries, texts)
add_documents(table_summaries, tables)
add_documents(image_summaries, img_list)

# 构建多向量检索（摘要索引）
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
    search_kwargs={"k": 7}
)

# 7.RAG构建
# 7.1 构建缩放图片存放目录
if os.path.exists(RESIZE_IMAGE_DIR):
    shutil.rmtree(RESIZE_IMAGE_DIR)
os.makedirs(RESIZE_IMAGE_DIR)


# 7.2 RAG构建
def prompt_func(data_dict):
    # "context":{"images": ["缩放后的图片地址","缩放后的图片地址"], "texts": "doc1"}
    # 提取图片与文本
    images = data_dict["context"]["images"]
    texts = data_dict["context"]["texts"]
    messages = []
    # 装载图片数据
    for image in images:
        messages.append(
            HumanMessage(
                content=[
                    {"text": f"请将图片标记标注为：`{image}`"},
                    {"image": f"file://{image}"}
                ]
            )
        )

    # 装载文本数据
    formatted_texts = "\n\n".join(texts)
    # 该提示词可优化，如加入少量样本，让大模型输出更稳定
    messages.append(
        HumanMessage(content=[
            {
                "text":
                    "你是作为一名专业的电气工程师和电路理论专家。你的任务是用中文回答与电路基本概念和定律相关的问题。"  # 角色
                    "你将获得相关的图片与文本作为参考的上下文。这些图片和文本都是根据用户输入的关键词从向量数据库中检索获取的。"
                    "请根据提供的图片和文本结合你丰富的知识与分析能力，提供一份全面的问题解答。"  # 任务
                    "请将提供的图片标记自然融入答案阐述的对应位置进行合理排版，答案中提及的图片统一以`<image>传递的图片真实标记</image>`的形式呈现。\n\n"  # 限制
                    f"用户的问题是：{data_dict['question']}\n\n"
                    "参考的文本或者表格数据：\n"
                    f"{formatted_texts}"
            }
        ]
        )
    )

    print('*' * 200)
    print(f"messages: {messages}")
    print('*' * 200)
    return messages


# 千问视觉模型（多模态）
llm = ChatTongyi(model="qwen-vl-max")
chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(split_image_text_types)
        }
        | RunnableLambda(prompt_func)
        | llm
        | StrOutputParser()
)

# 8. 效果展示
# 8.1 向量召回展示
query = "介绍下手电筒的电路模型"
docs = retriever.invoke(query)

print("1. 向量召回结果展示在jupyter中" + '-' * 100)
for doc in docs:
    if is_image_path(doc):
        show_plt_img(encode_image(doc))

# 8.2 RAG召回展示
result = chain.invoke(query)
print("2. RAG召回结果" + '-' * 100)
display_answer(result)
