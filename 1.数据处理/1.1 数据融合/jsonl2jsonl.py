# -*- coding: utf-8 -*-
"""
jsonl2jsonl.py - 处理jsonl文件，进行字段映射转换
"""
import json
import os


def jsonl2jsonl(input_path, output_path, field_mapping=None):
    """
    将jsonl文件进行字段映射转换
    
    Args:
        input_path: 输入jsonl文件路径
        output_path: 输出jsonl文件路径
        field_mapping: 字段映射关系字典，key为源字段，value为目标字段
                       例如: {'question': 'question', 'answer': 'answer', 'text': 'text'}
    
    Returns:
        处理的记录数
    """
    processed_count = 0
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                # 解析jsonl行
                input_record = json.loads(line)
                
                # 根据字段映射构建输出记录
                output_record = {}
                if field_mapping:
                    for source_field, target_field in field_mapping.items():
                        if source_field in input_record:
                            output_record[target_field] = input_record[source_field]
                else:
                    # 如果没有字段映射，保留所有字段
                    output_record = input_record
                
                # 写入jsonl文件
                outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                processed_count += 1
                
    except Exception as e:
        print(f"处理jsonl文件失败: {e}")
        return 0
    
    print(f"jsonl2jsonl处理完成，共处理 {processed_count} 条记录，输出到: {output_path}")
    return processed_count


if __name__ == '__main__':
    # 测试代码
    input_file = "c:/code/7-医疗大模型/qwen3-medical-rag/data/raw/test.jsonl"
    output_file = "c:/code/7-医疗大模型/qwen3-medical-rag/output/fusion/test.jsonl"
    field_mapping = {
        'question': 'question',
        'answer': 'answer',
        'text': 'text'
    }
    jsonl2jsonl(input_file, output_file, field_mapping)
