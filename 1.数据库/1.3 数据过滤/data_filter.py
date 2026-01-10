import os
import json
from pathlib import Path


def is_politically_sensitive(text: str) -> bool:
    """
    检测是否包含政治敏感内容
    
    Args:
        text: 待检测的文本内容
    
    Returns:
        bool: True表示包含政治敏感内容，False表示不包含
    """
    if not text or not text.strip():
        return False
    
    # 政治敏感关键词列表
    political_keywords = [
        "政治敏感", "政府", "选举", "领导人", "党派", "抗议", "示威", 
        "政权", "政党", "政治斗争", "政治危机", "政治改革"
    ]
    
    text_lower = text.lower()
    for keyword in political_keywords:
        if keyword in text_lower:
            return True
    
    return False


def is_pornographic_content(text: str) -> bool:
    """
    检测是否包含色情内容
    
    Args:
        text: 待检测的文本内容
    
    Returns:
        bool: True表示包含色情内容，False表示不包含
    """
    if not text or not text.strip():
        return False
    
    # 色情敏感关键词列表
    pornographic_keywords = [
        "色情", "裸体", "性交", "黄色", "成人", "激情", "诱惑",
        "色情片", "AV", "成人电影", "淫秽", "情色"
    ]
    
    text_lower = text.lower()
    for keyword in pornographic_keywords:
        if keyword in text_lower:
            return True
    
    return False


def is_target_language(text: str) -> bool:
    """
    检测是否为目标语言（中文）
    
    Args:
        text: 待检测的文本内容
    
    Returns:
        bool: True表示是目标语言，False表示不是
    """
    if not text or not text.strip():
        return False
    
    # 检查文本中中文字符的比例
    chinese_chars = 0
    total_chars = len(text)
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 基本汉字范围
            chinese_chars += 1
    
    # 如果中文字符占比超过30%，认为是目标语言
    if total_chars > 0 and chinese_chars / total_chars > 0.3:
        return True
    
    return False


def should_filter(text: str) -> bool:
    """
    判断是否应该过滤该文档
    
    Args:
        text: 文档文本内容
    
    Returns:
        bool: True表示应该过滤，False表示保留
    """
    # 检查是否包含政治敏感内容
    if is_politically_sensitive(text):
        return True
    
    # 检查是否包含色情内容
    if is_pornographic_content(text):
        return True
    
    # 检查是否为目标语言（如果不是则过滤）
    if not is_target_language(text):
        return True
    
    return False


def process_jsonl_file(input_path: str, output_path: str) -> dict:
    """
    处理单个jsonl文件，过滤不符合条件的文档
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    
    Returns:
        dict: 处理统计信息
    """
    stats = {
        "total": 0,
        "filtered": 0,
        "kept": 0,
        "filtered_by_political": 0,
        "filtered_by_pornographic": 0,
        "filtered_by_language": 0
    }
    
    kept_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                stats["total"] += 1
                
                # 获取文本内容（可能是text、content或question+answer字段）
                text = ""
                if "text" in data:
                    text = data["text"]
                elif "content" in data:
                    text = data["content"]
                elif "question" in data and "answer" in data:
                    text = data["question"] + " " + data["answer"]
                
                # 判断是否需要过滤
                if should_filter(text):
                    stats["filtered"] += 1
                    
                    # 统计过滤原因
                    if is_politically_sensitive(text):
                        stats["filtered_by_political"] += 1
                    elif is_pornographic_content(text):
                        stats["filtered_by_pornographic"] += 1
                    elif not is_target_language(text):
                        stats["filtered_by_language"] += 1
                else:
                    stats["kept"] += 1
                    kept_data.append(data)
                    
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                continue
    
    # 写入保留的文档
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in kept_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    return stats


def main():
    """
    主函数：处理output/clean目录下的所有文件
    """
    # 定义输入输出目录
    root_dir = Path(__file__).parent.parent.parent
    input_dir = root_dir / "output" / "clean"
    output_dir = root_dir / "output" / "filter"
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有jsonl文件
    jsonl_files = list(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"在 {input_dir} 目录下未找到jsonl文件")
        return
    
    total_stats = {
        "total": 0,
        "filtered": 0,
        "kept": 0,
        "filtered_by_political": 0,
        "filtered_by_pornographic": 0,
        "filtered_by_language": 0
    }
    
    # 处理每个文件
    for jsonl_file in jsonl_files:
        filename = jsonl_file.name
        output_path = output_dir / filename
        
        print(f"处理文件: {filename}")
        stats = process_jsonl_file(str(jsonl_file), str(output_path))
        
        print(f"  总文档数: {stats['total']}")
        print(f"  保留文档数: {stats['kept']}")
        print(f"  过滤文档数: {stats['filtered']}")
        print(f"    - 政治敏感: {stats['filtered_by_political']}")
        print(f"    - 色情内容: {stats['filtered_by_pornographic']}")
        print(f"    - 非目标语言: {stats['filtered_by_language']}")
        print()
        
        # 累计统计
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # 打印总体统计
    print("=" * 50)
    print("总体统计:")
    print(f"  总文档数: {total_stats['total']}")
    print(f"  保留文档数: {total_stats['kept']}")
    print(f"  过滤文档数: {total_stats['filtered']}")
    print(f"    - 政治敏感: {total_stats['filtered_by_political']}")
    print(f"    - 色情内容: {total_stats['filtered_by_pornographic']}")
    print(f"    - 非目标语言: {total_stats['filtered_by_language']}")
    print(f"  保留率: {total_stats['kept'] / total_stats['total'] * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
