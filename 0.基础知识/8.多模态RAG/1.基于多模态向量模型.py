import base64
import io
import os
import re
import shutil
import uuid
from typing import List

import dashscope
import markdown
from IPython.display import HTML, display
from PIL import Image
from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel
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


def show_plt_img(img_base64):
    """
    用于在 Jupyter Notebook 或类似环境中显示 Base64 编码的图像
    1. 使用 f-string 格式化创建一个 HTML 的 <img> 标签
    2. 标签的 src 属性使用 Data URL 格式:
    3. data:image/jpeg;base64, 表示这是一个 JPEG 图像的 Base64 编码数据
    4. 后面接上传入的 img_base64 字符串
    5. 使用 display(HTML(...)) 在 Notebook 中渲染这个 HTML 图像标签
    """
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))


def encode_image(img_path):
    """
    base64.b64encode(img_data) 将二进制数据编码为 Base64 字节串
    .decode('utf-8') 将 Base64 字节串转换为 UTF-8 字符串
    返回最终的 Base64 编码字符串
    """
    with open(img_path, "rb") as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return img_base64


def display_answer(text: str):
    """
    输入参数:
    text: 字符串类型，包含文本内容和图像标记的特殊格式字符串
    标记格式:使用 <image> 和 </image> 作为图像路径的标记
    格式示例: "这是一些文本<image>path/to/image.jpg</image>更多文本"
    """
    start_tag = "<image>"
    end_tag = "</image>"

    # 根据<image>标签 分割文本：xxx<image>xxx</image> => ['xxx','xxx</image>']
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
            display(HTML(markdown.markdown(context)))
        else:
            display(HTML(markdown.markdown(part)))


def resize_base64_image4tongyi(base64_string, max_size=(640, 480)):
    '''
    将Base64编码的图片进行缩放处理，并将缩放后的图片保存到本地文件系统，最后返回保存路径
    base64_string: Base64编码的图片字符串
        它是一个字符串，包含字母(A-Z, a-z)、数字(0-9)以及特殊字符(+/=)
        通常以类似"data:image/png;base64,"开头，后面跟着实际的Base64编码数据
        例如： "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8//+/JgMRQBOgLN4aAKkDZQ0V6XQZAAAAAElFTkSuQmCC"
    max_size: 一个元组，表示图片缩放后的最大尺寸，默认为(640, 480)
    '''
    # 解析图片
    img_data = base64.b64decode(base64_string)  # 将Base64字符串解码为二进制图片数据
    img = Image.open(io.BytesIO(img_data))  # 使用PIL库的Image.open()和io.BytesIO()从二进制数据创建图片对象

    width, height = img.size
    ratio = max(max_size[0] / width, max_size[1] / height)

    # 计算按比例缩放后的宽高
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # 缩放 Image.LANCZOS: 重采样滤波器
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # 保存图片到指定路径
    out_path = os.path.join(RESIZE_IMAGE_DIR, str(uuid.uuid4()) + ".jpg")
    resized_img.save(out_path)
    # 图片地址
    return out_path


def is_base64(s):
    """检查是否为base64数据"""
    # 检查是否为字符串
    if not isinstance(s, str):
        return False

    # 检查是否只包含 base64 允许的字符
    if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', s):
        return False

    # 检查长度是否是 4 的倍数
    if len(s) % 4 != 0:
        return False

    # 尝试解码
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def split_image_text_types(docs):
    """
    将文档和图片内容切分，分别存储到各自的列表中
    """
    images = []
    text = []
    for doc in docs:
        content = doc.page_content
        if is_base64(content):
            # 缩放图片
            resize_image = resize_base64_image4tongyi(content)
            images.append(resize_image)
        else:
            text.append(content)

    return {"images": images, "texts": text}


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

print("pdf_data 格式：[CompositeElement ，table，CompositeElement ，table,...]: ")
print(pdf_data)
print("查看 CompositeElement" + "-" * 100)
print(pdf_data[0].metadata.orig_elements)
print("查看 CompositeElement 的子节点" + "-" * 100)
print(pdf_data[0].metadata.orig_elements[1])

# 3. 提取表格与文本
tables = []
texts = []
for element in pdf_data:
    if isinstance(element, Table):
        tables.append(str(element))
    elif isinstance(element, CompositeElement):
        texts.append(str(element))
print(f"表格元素：{len(tables)}  文本元素：{len(texts)}")


# 4. 多模态嵌入模型
class MultiDashScopeEmbeddings(BaseModel, Embeddings):
    model: str = "multimodal-embedding-v1"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_features = []
        for text in texts:
            resp = dashscope.MultiModalEmbedding.call(model=self.model, input=[{'text': text}])
            while resp.output is None:
                print(f"{text} 向量化失败！")
                resp = dashscope.MultiModalEmbedding.call(model=self.model, input=[{'text': text}])

            embeddings_list = resp.output['embeddings'][0]['embedding']
            text_features.append(embeddings_list)
        return text_features

    def embed_query(self, text: str) -> List[float]:
        resp = dashscope.MultiModalEmbedding.call(model=self.model, input=[{'text': text}])
        while resp.output is None:
            print(f"{text} 向量化失败！")
            resp = dashscope.MultiModalEmbedding.call(model=self.model, input=[{'text': text}])

        embeddings_list = resp.output['embeddings'][0]['embedding']
        return embeddings_list

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        image_features = []
        for uri in uris:
            # 阿里dashscope SDK要求传递图片的地址，对于本地图片dashscope SDK会将图片上传到OSS服务中：
            local_image_uri = f"file://{uri}"
            resp = dashscope.MultiModalEmbedding.call(model=self.model, input=[{"image": local_image_uri}])
            while resp.output is None:
                print(f"{local_image_uri} 向量化失败！")
                resp = dashscope.MultiModalEmbedding.call(model=self.model, input=[{"image": local_image_uri}])

            embeddings_list = resp.output['embeddings'][0]['embedding']
            image_features.append(embeddings_list)
        return image_features


# 5. 嵌入图片与文本
# 5.1 向量数据库
vectorstore = Chroma(collection_name="multi-vector", embedding_function=MultiDashScopeEmbeddings())

# 5.2 获得图片的地址
image_uris = sorted(
    [
        os.path.join(IMAGE_OUT_DIR, image_name)
        for image_name in os.listdir(IMAGE_OUT_DIR)
        if image_name.endswith(".jpg")
    ]
)

# 5.3 添加图片 （存储图像base64数据与其向量数据）
print("向量化图片" + '-' * 100)
if image_uris:
    vectorstore.add_images(uris=image_uris)  # embedding_function 向量化
# 5.4 添加文本与表格
print("向量化表格" + '-' * 100)
if tables:
    vectorstore.add_documents([Document(page_content=table) for table in tables])  # embedding_function 向量化
print("向量化文本" + '-' * 100)
if texts:
    vectorstore.add_documents([Document(page_content=text) for text in texts])
print("数据添加完毕!!!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 5.5 向量召回测试
query = "手电筒的电路模型"
docs = retriever.invoke(query)
print("1. 向量召回结果数： " + '-' * 100)
print(len(docs))
print("jupyter中展示图片：")
for doc in docs:
    if is_base64(doc.page_content):
        show_plt_img(doc.page_content)

# 6. RAG构建
# 6.1 缩放图片目录构建
if os.path.exists(RESIZE_IMAGE_DIR):
    shutil.rmtree(RESIZE_IMAGE_DIR)
os.makedirs(RESIZE_IMAGE_DIR)


# 6.2 提示构造函数
def prompt_func(data_dict):
    # "context":{"images": ["缩放后的图片地址","缩放后的图片地址"], "texts": "doc1"}
    # 提取图片与文本
    images = data_dict["context"]["images"]
    texts = data_dict["context"]["texts"]
    messages = []
    # 装载图片数据
    for image in images:
        # 样例：
        # HumanMessage(content=[{'text': '请将图片标记标注为：`C:\\...\\35589c8d-e802-4f7b-a647-41ed1a0a3439.jpg`'},
        # {'image': 'file://C:\\...\\35589c8d-e802-4f7b-a647-41ed1a0a3439.jpg'}],
        # additional_kwargs={},
        # response_metadata={}
        # )
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
    messages.append(
        # 样例：
        # HumanMessage(content=[{'text': '你是作为一名专...理排版，答案中提及的图片统一以`<image>传递的图片真实标记</image>`的形式呈现。\n
        # 用户的问题是：手电筒的电路模型\n\n参考的文本或者表格数据：\n'}],
        # additional_kwargs={},
        # response_metadata={})]
        HumanMessage(content=[
            {
                "text":
                    "你是作为一名专业的电气工程师和电路理论专家。你的任务是用中文回答与电路基本概念和定律相关的问题。"  # 角色
                    "你将获得相关的图片与文本作为参考的上下文。这些图片和文本都是根据用户输入的关键词从向量数据库中检索获取的。"
                    "请根据提供的图片和文本结合你丰富的知识与分析能力，提供一份全面的问题解答。"  # 任务
                    "请将提供的图片标记自然融入答案阐述的对应位置进行合理排版，答案中提及的图片统一以`<image>传递的图片真实标记</image>`的形式呈现。\n\n"  # 需求
                    f"用户的问题是：{data_dict['question']}\n\n"
                    "参考的文本或者表格数据：\n"  # 样例
                    f"{formatted_texts}"
            }
        ]
        )
    )

    print('*' * 200)
    print(f"messages: {messages}")
    print('*' * 200)
    return messages


# 6.3. 构建提示链
chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(split_image_text_types)
        } | RunnableLambda(prompt_func)
)
result = chain.invoke(query)
print("2. 提示结果：" + '-' * 100)
print(result)

# 6.3 构建RAG链
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

result = chain.invoke(query)
print('3. RAG结果:' + '-' * 100)
print(result)

print("jupyter中展示结果:" + '-' * 100)
display_answer(result)

"""
核心知识点：
1.
"""