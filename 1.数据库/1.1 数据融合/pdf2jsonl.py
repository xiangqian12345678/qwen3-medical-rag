# -*- coding: utf-8 -*-
"""
pdf2jsonl.py - 处理pdf文件，转换为jsonl格式
"""
import json
import os


def pdf2jsonl(input_path, output_path, field_mapping=None):
    """
    将pdf文件转换为jsonl格式
    
    Args:
        input_path: 输入pdf文件路径
        output_path: 输出jsonl文件路径
        field_mapping: 字段映射关系字典（pdf一般只有text字段，此参数保留用于扩展）
    
    Returns:
        处理的记录数
    """
    processed_count = 0
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # 尝试导入pdf处理库
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(input_path)
            
            # 提取所有文本
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
        except ImportError:
            print("未安装PyPDF2，尝试使用pypdf...")
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(input_path)
                
                # 提取所有文本
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                    
            except ImportError:
                print("未安装pdf处理库，请安装PyPDF2或pypdf: pip install PyPDF2")
                return 0
        
        # 构建输出记录
        if field_mapping and 'text' in field_mapping:
            target_field = field_mapping['text']
        else:
            target_field = 'text'
        
        output_record = {target_field: text_content}
        
        # 写入jsonl文件
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            processed_count = 1
            
    except Exception as e:
        print(f"处理pdf文件失败: {e}")
        return 0
    
    print(f"pdf2jsonl处理完成，共处理 {processed_count} 条记录，输出到: {output_path}")
    return processed_count


if __name__ == '__main__':
    # 测试代码
    input_file = "c:/code/7-医疗大模型/qwen3-medical-rag/data/raw/test.pdf"
    output_file = "c:/code/7-医疗大模型/qwen3-medical-rag/output/fusion/test.jsonl"
    pdf2jsonl(input_file, output_file)
