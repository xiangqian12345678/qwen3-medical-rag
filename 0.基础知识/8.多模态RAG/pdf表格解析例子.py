import os

from IPython.display import HTML, display
from unstructured.partition.pdf import partition_pdf

# 1. 初始化变量
# 请将此路径替换为你的目录
BASE_DIR = "C:\\大模型\\智泊大模型教程10期\\02-AGI大模型全栈班-L2阶段\\09-Advanced-RAG1，Pre-Retrieval预检索优化-25.4.15\\课件资料\\source"
# 资源地址
RESOURCE_DIR = BASE_DIR + '\\resources'
# 工具地址
TOOLS_DIR = BASE_DIR + '\\tools'

# 2. 环境配置
os.environ['PATH'] += os.pathsep + os.path.join(TOOLS_DIR, 'poppler-24.08.0/Library/bin')
TESSERACT_DIR = 'C:\\Program Files'
os.environ['PATH'] += os.pathsep + os.path.join(TESSERACT_DIR, 'Tesseract-OCR')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

pdf_data2 = partition_pdf(
    filename=r"C:\大模型\智泊大模型教程10期\02-AGI大模型全栈班-L2阶段\09-Advanced-RAG1，Pre-Retrieval预检索优化-25.4.15\课件资料\source\resources\MethodsPLCMOS12.pdf",
    infer_table_structure=True,  # 启用表格结构识别
    max_characters=4000,  # 每个文本块最大字符数
    new_after_n_chars=3800,  # 达到3800字符后分新块
    combine_text_under_n_chars=2000,  # 合并小于2000字符的文本块
    chunking_strategy="by_title",  # 按标题分块
    # 如果表格是中文，需要下载语言包：https://github.com/tesseract-ocr/tessdata
    # 放入Tesseract安装目录下的tessdata文件夹
    languages=["chi_sim"]
)

element = pdf_data2[0]
print(element.text)
print(pdf_data2[0].metadata.text_as_html)
display(HTML(pdf_data2[0].metadata.text_as_html))
