"""
数据标注模块
根据annotation.yaml配置对数据进行chunk切分和标注
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Generator

import yaml


def load_annotation_config(config_path: str) -> Dict[str, Any]:
    """加载annotation.yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_keyword() -> List[str]:
    """关键词生成函数，默认返回空列表"""
    return []


def generate_questions() -> List[str]:
    """问题生成函数，默认返回空列表"""
    return []


def generate_summary() -> str:
    """摘要生成函数，默认返回空字符串"""
    return ""


def create_chunks_with_parent(text: str, chunk_size: int, overlap: int, parent_size: int) -> Generator[
    tuple[int, str, str], None, None]:
    """
    将文本切分成chunk，同时计算对应的parent_chunk

    Args:
        text: 原始文本
        chunk_size: chunk大小
        overlap: 重叠大小
        parent_size: 父chunk大小

    Yields:
        (chunk_id, chunk_text, parent_chunk_text) 元组
    """
    text_len = len(text)

    if text_len <= chunk_size:
        yield (1, text, text)
        return

    # 预先生成所有子chunks（只保存，不yield）
    chunks_list = []
    start = 0
    prev_start = -1  # 记录上一次的start位置，用于检测无限循环

    while start < text_len:
        # 检测是否陷入无限循环
        if start == prev_start:
            break

        prev_start = start
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks_list.append(chunk)

        # 更新起始位置，考虑重叠
        # 使用 step 确保 start 至少前进 overlap 大小
        start = end - overlap

        # 防止重叠导致无限循环
        if start <= 0 or start >= end:
            start = end

    # 逐个计算parent_chunk并立即yield，不保存parent_chunk
    for idx, chunk in enumerate(chunks_list):
        # 计算当前chunk对应的parent_chunk
        total_length = 0
        included_chunks = []

        # 从当前chunk开始向前收集
        for i in range(idx, -1, -1):
            chunk_len = len(chunks_list[i])
            if total_length + chunk_len > parent_size:
                break
            included_chunks.insert(0, chunks_list[i])
            total_length += chunk_len

        # 向后继续收集
        for i in range(idx + 1, len(chunks_list)):
            if total_length >= parent_size:
                break
            chunk_len = len(chunks_list[i])
            if total_length + chunk_len > parent_size:
                break
            included_chunks.append(chunks_list[i])
            total_length += chunk_len

        parent_chunk = "".join(included_chunks)

        # yield后立即释放parent_chunk引用
        yield (idx + 1, chunk, parent_chunk)
        del parent_chunk


def get_parent_chunk_for_child(chunks: List[str], chunk_size: int, child_index: int) -> str:
    """
    根据子chunk获取对应的父chunk
    
    Args:
        chunks: 所有子chunk列表
        chunk_size: 父chunk大小
        child_index: 子chunk索引
    
    Returns:
        父chunk文本
    """
    # 计算需要包含的子chunk数量
    total_length = 0
    included_chunks = []

    # 从当前chunk开始向前收集，直到达到父chunk大小
    for i in range(child_index, -1, -1):
        chunk_len = len(chunks[i])
        if total_length + chunk_len > chunk_size:
            break
        included_chunks.insert(0, chunks[i])
        total_length += chunk_len

    # 如果还没有达到父chunk大小，向后继续收集
    for i in range(child_index + 1, len(chunks)):
        if total_length >= chunk_size:
            break
        chunk_len = len(chunks[i])
        if total_length + chunk_len > chunk_size:
            break
        included_chunks.append(chunks[i])
        total_length += chunk_len

    return "".join(included_chunks)


def annotate_record(record: Dict[str, Any], config: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    对单条记录进行标注

    Args:
        record: 原始记录
        config: 配置字典

    Yields:
        标注后的chunk字典
    """
    # 获取配置
    chunk_size = config.get('chunk_size', 512)
    overlap = config.get('overlap', 0)
    parent_chunk_config = config.get('parent_chunk', {})
    summary_config = config.get('summary', {})
    questions_config = config.get('questions', {})
    keyword_config = config.get('keyword', {})
    chunk_id_config = config.get('chunk_id', {})
    preserve_fields = config.get('preserve_fields', {})

    # 获取text字段并立即释放原始record引用（如果text在record中）
    text = record.get('text', '')
    if not text:
        return

    # 提取需要保留的其他字段（避免后续访问record）
    preserved_data = {}
    for field, should_preserve in preserve_fields.items():
        if should_preserve and field != 'text' and field in record:
            preserved_data[field] = record[field]

    # 如果配置不保留原始text，立即释放text引用
    if not preserve_fields.get('text', False):
        pass  # Python会自动处理

    parent_size = parent_chunk_config.get('chunk_size', 2048) if parent_chunk_config.get('enabled', False) else 0

    # 使用生成器逐个处理chunk
    if parent_chunk_config.get('enabled', False):
        # 需要parent_chunk的情况
        for chunk_id, chunk, parent_chunk in create_chunks_with_parent(text, chunk_size, overlap, parent_size):
            annotated = {'chunk': chunk, 'parent_chunk': parent_chunk}

            # 添加其他字段
            if summary_config.get('enabled', False):
                annotated['summary'] = generate_summary()

            if questions_config.get('enabled', False):
                annotated['questions'] = generate_questions()

            if keyword_config.get('enable', False):
                annotated['keyword'] = generate_keyword()

            if chunk_id_config.get('enabled', False):
                annotated['chunk_id'] = chunk_id

            # 添加保留字段
            annotated.update(preserved_data)

            # 显式释放引用
            del parent_chunk, chunk
            yield annotated
    else:
        # 不需要parent_chunk的情况，完全流式处理
        start = 0
        chunk_id = 1
        text_len = len(text)
        prev_start = -1  # 用于检测无限循环

        while start < text_len:
            # 检测是否陷入无限循环
            if start == prev_start:
                break

            prev_start = start
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]

            annotated = {'chunk': chunk}

            if summary_config.get('enabled', False):
                annotated['summary'] = generate_summary()

            if questions_config.get('enabled', False):
                annotated['questions'] = generate_questions()

            if keyword_config.get('enable', False):
                annotated['keyword'] = generate_keyword()

            if chunk_id_config.get('enabled', False):
                annotated['chunk_id'] = chunk_id

            # 添加保留字段
            annotated.update(preserved_data)

            # 更新起始位置，考虑重叠
            start = end - overlap
            if start <= 0 or start >= end:
                start = end

            chunk_id += 1

            del chunk
            yield annotated


def read_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    读取jsonl格式文件
    
    Args:
        file_path: 文件路径
    
    Yields:
        解析后的记录字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(file_path: str, records: List[Dict[str, Any]]) -> None:
    """
    写入jsonl格式文件
    
    Args:
        file_path: 文件路径
        records: 记录列表
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def annotate_file(
        input_file: str,
        output_file: str,
        config_path: str
) -> None:
    """
    对jsonl文件进行标注处理

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        config_path: 配置文件路径
    """
    import gc

    # 加载配置
    config = load_annotation_config(config_path)

    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 清空目标文件（如果存在）
    open(output_file, 'w', encoding='utf-8').close()

    # 每处理一条数据就追加到目标文件
    linecount = 1
    with open(output_file, 'a', encoding='utf-8') as f:
        for record in read_jsonl(input_file):
            print(f"处理第{linecount}条数据")
            linecount += 1

            # 使用生成器逐个处理chunk，避免在内存中累积
            for chunk in annotate_record(record, config):
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

            # 显式释放记录引用，触发垃圾回收
            del record

def main():
    """主函数"""
    # 定义路径
    project_root = Path(__file__).parent.parent.parent  # 工程根目录
    input_dir = project_root / 'output' / 'filter'
    output_dir = project_root / 'output' / 'annotation'
    config_dir = Path(__file__).parent
    config_file = config_dir / 'annotation.yaml'

    # 处理filter目录下的所有jsonl文件
    for input_file in input_dir.glob('*.jsonl'):
        if input_file.stat().st_size == 0:
            continue

        output_file = output_dir / f"{input_file.stem}.jsonl"
        print(f"处理文件: {input_file.name}")

        try:
            annotate_file(str(input_file), str(output_file), str(config_file))
            print(f"输出文件: {output_file.name}")
        except Exception as e:
            print(f"处理文件 {input_file.name} 时出错: {e}")


if __name__ == '__main__':
    main()
