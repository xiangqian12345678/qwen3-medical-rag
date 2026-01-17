"""
文本处理模块
负责从文件读取文本和预处理
"""
import os
from typing import List
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader,
    UnstructuredMarkdownLoader, UnstructuredFileLoader
)


class TextProcessor:
    """
    文本处理器类
    负责从各种文件格式加载文本
    """

    def __init__(self):
        """
        初始化文本处理器

        【输入示例】
        processor = TextProcessor()

        【输出示例】
        None (处理器已初始化)
        """
        self.supported_extensions = ['.txt', '.pdf', '.docx', '.doc', '.md']

    def load_text_from_file(self, file_path: str) -> List[str]:
        """
        从文件加载文本

        【输入示例】
        texts = processor.load_text_from_file("documents/demo.txt")

        【输出示例】
        ["阿司匹林是一种非甾体抗炎药...", "主要用于治疗头痛、发热..."]
        """
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return []

        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            # 根据文件类型选择加载器
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path)
            else:
                # 尝试使用通用加载器
                loader = UnstructuredFileLoader(file_path)

            # 加载文档
            docs = loader.load()

            # 提取文本内容
            texts = []
            for doc in docs:
                if doc.page_content and doc.page_content.strip():
                    texts.append(doc.page_content.strip())

            print(f"✅ 从 {file_path} 加载了 {len(texts)} 段文本")
            return texts

        except Exception as e:
            print(f"❌ 加载文件失败 {file_path}: {e}")
            return []

    def load_text_from_directory(self, directory: str,
                                   extensions: List[str] = None) -> List[str]:
        """
        从目录加载所有文本

        【输入示例】
        texts = processor.load_text_from_directory(
            "documents/",
            extensions=['.txt', '.docx']
        )

        【输出示例】
        ["文本1...", "文本2...", "文本3..."]
        """
        extensions = extensions or self.supported_extensions

        all_texts = []

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext in extensions:
                texts = self.load_text_from_file(file_path)
                all_texts.extend(texts)

        print(f"✅ 从目录 {directory} 共加载 {len(all_texts)} 段文本")
        return all_texts

    def split_text(self, text: str, chunk_size: int = 1000,
                   overlap: int = 100) -> List[str]:
        """
        分割长文本为短片段

        【输入示例】
        chunks = processor.split_text(
            long_text,
            chunk_size=500,
            overlap=50
        )

        【输出示例】
        ["片段1...", "片段2...", "片段3..."]
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # 避免在单词中间分割
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > 0:
                    chunk = chunk[:last_space]
                    end = start + last_space + 1

            chunks.append(chunk.strip())
            start = end - overlap

            if len(chunks[-1]) == 0:
                break

        return [c for c in chunks if c]

    def load_and_split_file(self, file_path: str,
                             chunk_size: int = 1000,
                             overlap: int = 100) -> List[str]:
        """
        加载文件并分割文本

        【输入示例】
        chunks = processor.load_and_split_file(
            "documents/demo.txt",
            chunk_size=500,
            overlap=50
        )

        【输出示例】
        ["片段1...", "片段2...", "片段3..."]
        """
        texts = self.load_text_from_file(file_path)

        all_chunks = []
        for text in texts:
            chunks = self.split_text(text, chunk_size, overlap)
            all_chunks.extend(chunks)

        print(f"✅ 文件处理完成，共生成 {len(all_chunks)} 个片段")
        return all_chunks


# 使用示例
if __name__ == "__main__":
    # 示例1: 加载单个文件
    print("示例1: 加载单个文件")
    processor = TextProcessor()

    # 假设存在测试文件
    test_file = "../data/documents/demo.txt"
    if os.path.exists(test_file):
        texts = processor.load_text_from_file(test_file)
        print(f"加载了 {len(texts)} 段文本")
        if texts:
            print(f"第一段: {texts[0][:100]}...")

    # 示例2: 加载整个目录
    print("\n示例2: 加载整个目录")
    if os.path.exists("../data/documents"):
        texts = processor.load_text_from_directory("documents")
        print(f"总共加载了 {len(texts)} 段文本")

    # 示例3: 分割文本
    print("\n示例3: 分割文本")
    long_text = "阿司匹林是一种非甾体抗炎药，具有镇痛、解热、抗炎作用。它通过抑制环氧合酶，减少前列腺素的合成，从而发挥药理作用。阿司匹林广泛用于治疗头痛、关节痛、肌肉痛、牙痛等轻中度疼痛，以及感冒引起的发热。此外，小剂量阿司匹林还具有抗血小板聚集作用，可用于预防心脑血管疾病。"
    chunks = processor.split_text(long_text, chunk_size=100, overlap=20)
    print(f"分割成 {len(chunks)} 个片段:")
    for i, chunk in enumerate(chunks):
        print(f"  片段{i+1}: {chunk[:50]}...")

    # 示例4: 加载并分割
    print("\n示例4: 加载并分割文件")
    if os.path.exists(test_file):
        chunks = processor.load_and_split_file(test_file, chunk_size=200)
        print(f"生成了 {len(chunks)} 个片段")
