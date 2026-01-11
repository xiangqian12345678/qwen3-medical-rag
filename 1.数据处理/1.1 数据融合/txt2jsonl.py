# -*- coding: utf-8 -*-
"""
txt2jsonl.py - 处理txt文件，转换为jsonl格式
"""
import json
import os


def txt2jsonl(input_path, output_path, field_mapping=None):
    """
    将txt文件转换为jsonl格式
    
    Args:
        input_path: 输入txt文件路径
        output_path: 输出jsonl文件路径
        field_mapping: 字段映射关系字典（txt一般只有text字段，此参数保留用于扩展）
    
    Returns:
        处理的记录数
    """
    processed_count = 0
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            # 读取整个txt内容作为一个text字段
            content = infile.read()
            
            # 构建输出记录
            if field_mapping and 'text' in field_mapping:
                target_field = field_mapping['text']
            else:
                target_field = 'text'
            
            output_record = {target_field: content}
            
            # 写入jsonl文件
            outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            processed_count = 1
            
    except Exception as e:
        print(f"处理txt文件失败: {e}")
        return 0
    
    print(f"txt2jsonl处理完成，共处理 {processed_count} 条记录，输出到: {output_path}")
    return processed_count


if __name__ == '__main__':
    # 测试代码
    input_file = "c:/code/7-医疗大模型/qwen3-medical-rag/data/raw/test.txt"
    output_file = "c:/code/7-医疗大模型/qwen3-medical-rag/output/fusion/test.jsonl"
    txt2jsonl(input_file, output_file)
