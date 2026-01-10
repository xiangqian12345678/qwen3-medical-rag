# -*- coding: utf-8 -*-
"""
dataFusion.py - 数据融合主程序
根据配置文件处理不同类型的数据文件
"""
import yaml
import os
import sys

# 添加父目录到路径，以便导入同级模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from txt2jsonl import txt2jsonl
from pdf2jsonl import pdf2jsonl
from jsonl2jsonl import jsonl2jsonl


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_field_mapping(file_config):
    """
    从配置中提取字段映射关系
    
    Args:
        file_config: 单个文件的配置字典
    
    Returns:
        字段映射字典
    """
    field_mapping = {}
    for key, value in file_config.items():
        if key != 'type':
            field_mapping[key] = value
    return field_mapping


def process_file(file_name, file_config, project_root):
    """
    处理单个文件
    
    Args:
        file_name: 文件名（不含扩展名）
        file_config: 文件配置
        project_root: 工程根目录
    
    Returns:
        处理的记录数
    """
    file_type = file_config.get('type')
    
    # 根据文件类型确定输入文件路径
    if file_type == 'jsonl':
        input_file = os.path.join(project_root, 'data', 'raw', f'{file_name}.jsonl')
    elif file_type == 'txt':
        input_file = os.path.join(project_root, 'data', 'raw', f'{file_name}.txt')
    elif file_type == 'pdf':
        input_file = os.path.join(project_root, 'data', 'raw', f'{file_name}.pdf')
    else:
        print(f"不支持的文件类型: {file_type}")
        return 0
    
    # 输出文件路径
    output_file = os.path.join(project_root, 'output', 'fusion', f'{file_name}.jsonl')
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return 0
    
    # 获取字段映射关系
    field_mapping = get_field_mapping(file_config)
    
    # 根据文件类型调用对应处理函数
    processed_count = 0
    if file_type == 'jsonl':
        print(f"处理jsonl文件: {input_file}")
        processed_count = jsonl2jsonl(input_file, output_file, field_mapping)
    elif file_type == 'txt':
        print(f"处理txt文件: {input_file}")
        processed_count = txt2jsonl(input_file, output_file, field_mapping)
    elif file_type == 'pdf':
        print(f"处理pdf文件: {input_file}")
        processed_count = pdf2jsonl(input_file, output_file, field_mapping)
    
    return processed_count


def main():
    # 获取配置文件路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'fusion.yaml')
    
    # 获取工程根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print(f"工程根目录: {project_root}")
    print(f"配置文件路径: {config_path}")
    
    # 加载配置
    config = load_config(config_path)
    print(f"加载配置文件成功，共 {len(config)} 个文件需要处理")
    
    # 处理每个文件
    total_processed = 0
    for file_name, file_config in config.items():
        print(f"\n{'='*50}")
        print(f"开始处理文件: {file_name}")
        print(f"{'='*50}")
        
        processed_count = process_file(file_name, file_config, project_root)
        total_processed += processed_count
    
    print(f"\n{'='*50}")
    print(f"所有文件处理完成，共处理 {total_processed} 条记录")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
