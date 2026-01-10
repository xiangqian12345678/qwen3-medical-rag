import json
import os
from pathlib import Path
from typing import Dict, Any


def check_coherence(document: Dict[str, Any]) -> bool:
    """
    连贯性判断
    检查文档内容的逻辑连贯性和完整性
    
    Args:
        document: 文档字典对象
    
    Returns:
        bool: 默认返回True，将来可添加具体的连贯性判断逻辑
    """
    # TODO: 实现具体的连贯性判断逻辑
    # 示例：检查文档长度、结构完整性等
    return True


def check_practicality(document: Dict[str, Any]) -> bool:
    """
    实用性判断
    检查文档内容的实用价值和可操作性
    
    Args:
        document: 文档字典对象
    
    Returns:
        bool: 默认返回True，将来可添加具体的实用性判断逻辑
    """
    # TODO: 实现具体的实用性判断逻辑
    # 示例：检查是否包含可操作的建议、是否是重复内容等
    return True


def check_safety(document: Dict[str, Any]) -> bool:
    """
    安全性判断
    检查文档内容的安全性，避免有害信息
    
    Args:
        document: 文档字典对象
    
    Returns:
        bool: 默认返回True，将来可添加具体的安全性判断逻辑
    """
    # TODO: 实现具体的安全性判断逻辑
    # 示例：检查是否包含敏感信息、错误医疗建议等
    return True


def check_cleanliness(document: Dict[str, Any]) -> bool:
    """
    清洗度判断
    检查文档的基本清洗质量
    
    Args:
        document: 文档字典对象
    
    Returns:
        bool: 默认返回True，将来可添加具体的清洗度判断逻辑
    """
    # TODO: 实现具体的清洗度判断逻辑
    # 示例：检查字段完整性、数据格式等
    return True


def validate_document(document: Dict[str, Any]) -> bool:
    """
    综合验证文档质量
    所有判断函数都通过才返回True
    
    Args:
        document: 文档字典对象
    
    Returns:
        bool: 是否通过所有质量检查
    """
    # 依次执行各项质量检查
    if not check_coherence(document):
        return False
    
    if not check_practicality(document):
        return False
    
    if not check_safety(document):
        return False
    
    if not check_cleanliness(document):
        return False
    
    return True


def process_jsonl_file(input_path: Path, output_path: Path) -> int:
    """
    处理单个jsonl文件
    读取、验证并保存通过质量检查的文档
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    
    Returns:
        int: 通过验证的文档数量
    """
    passed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                document = json.loads(line)
                
                # 验证文档质量
                if validate_document(document):
                    outfile.write(json.dumps(document, ensure_ascii=False) + '\n')
                    passed_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 {input_path.name}:{line_num}: {e}")
                continue
    
    return passed_count


def main():
    """
    主函数：处理output/fusion/目录下的所有文件
    """
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "output" / "fusion"
    output_dir = project_root / "output" / "clean"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保输入目录存在
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return
    
    # 处理所有jsonl文件
    total_passed = 0
    processed_files = 0
    
    for input_file in sorted(input_dir.glob("*.jsonl")):
        output_file = output_dir / input_file.name
        
        print(f"正在处理: {input_file.name}")
        passed = process_jsonl_file(input_file, output_file)
        
        total_passed += passed
        processed_files += 1
        
        print(f"  通过验证: {passed} 条记录")
    
    # 输出统计信息
    print(f"\n========== 处理完成 ==========")
    print(f"处理文件数: {processed_files}")
    print(f"总通过记录数: {total_passed}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
