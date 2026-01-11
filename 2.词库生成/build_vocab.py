"""词库构建主程序 - 从标注数据构建BM25词表"""

import glob
import time
from pathlib import Path

from datasets import load_dataset

from vocabulary import Vocabulary
from tokenizer import Tokenizer


def load_jsonl_files(input_dir: str) -> list:
    """加载输入目录下的所有 JSONL 文件"""
    input_path = Path(input_dir)
    file_pattern = str(input_path / "*.jsonl")
    files = sorted(glob.glob(file_pattern))

    if not files:
        raise FileNotFoundError(f"未找到 JSONL 文件: {file_pattern}")

    print(f"找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {f}")

    return files


def build_vocab_from_files(
    input_dir: str,
    output_dir: str,
    domain_model: str = "medicine",
    workers: int = 8,
    chunksize: int = 128
) -> None:
    """
    从输入目录的 JSONL 文件构建词表

    Args:
        input_dir: 输入文件目录
        output_dir: 输出文件目录
        domain_model: 领域模型名称
        workers: 并行工作进程数
        chunksize: 每批处理的文档数
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找输入文件
    files = load_jsonl_files(input_dir)

    # 初始化词表和分词器
    vocab = Vocabulary()
    tokenizer = Tokenizer(domain_model=domain_model)

    # 开始构建
    t0 = time.time()
    total_len = 0
    doc_count = 0

    print("\n开始构建词表...")

    for file_path in files:
        print(f"\n处理文件: {Path(file_path).name}")

        # 加载数据集
        dataset = load_dataset("json", data_files=file_path, split="train")

        # 确定文本字段
        if "chunk" in dataset.column_names:
            texts = dataset["chunk"]

        # 并行分词并构建词表
        for tokens in tokenizer.tokenize_parallel(texts, workers=workers, chunksize=chunksize):
            if tokens:
                vocab.add_document(tokens)
                total_len += len(tokens)
                doc_count += 1

                if doc_count % 10000 == 0:
                    print(f"  已处理 {doc_count} 个文档...")

    t1 = time.time()

    # 冻结词表
    vocab.freeze()

    # 保存词表
    output_file = str(output_path / "vocab.pkl.gz")
    vocab.save(output_file)

    # 输出统计信息
    print(f"\n{'='*50}")
    print(f"词表构建完成！")
    print(f"构建用时: {t1 - t0:.2f} 秒")
    print(f"文档总数: {vocab.total_docs}")
    print(f"总token数: {total_len:,}")
    print(f"词表大小: {len(vocab.token_to_id):,}")
    print(f"平均文档长度: {vocab.sum_doc_length / vocab.total_docs:.2f}")
    print(f"输出文件: {output_file}")
    print(f"{'='*50}\n")


def main():
    """主函数"""
    # 配置参数
    project_root = Path(__file__).parent.parent  # 工程根目录
    INPUT_DIR = project_root / 'output/annotation'
    OUTPUT_DIR = project_root / 'output/vocab'
    DOMAIN_MODEL = "medicine"
    WORKERS = 8
    CHUNKSIZE = 128

    build_vocab_from_files(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        domain_model=DOMAIN_MODEL,
        workers=WORKERS,
        chunksize=CHUNKSIZE
    )


if __name__ == "__main__":
    main()
