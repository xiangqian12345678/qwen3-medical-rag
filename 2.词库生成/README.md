# 词库生成模块

## 功能说明

本模块用于从标注数据构建BM25检索所需的词表。

## 文件结构

```
2.词库生成/
├── vocabulary.py   # 词表管理类
├── tokenizer.py    # 分词器（支持并行）
├── build_vocab.py  # 词库构建主程序
└── README.md       # 说明文档
```

## 使用方法

### 基本用法

```bash
cd c:/code/7-医疗大模型/qwen3-medical-rag/2.词库生成
python build_vocab.py
```

### 自定义参数

修改 `build_vocab.py` 中的配置：

```python
INPUT_DIR = "output/annotation"    # 输入目录
OUTPUT_DIR = "output/vocab"        # 输出目录
DOMAIN_MODEL = "medicine"          # 领域模型
WORKERS = 8                       # 并行进程数
CHUNKSIZE = 128                   # 每批处理文档数
```

## 输入输出

### 输入
- 目录：`output/annotation`
- 文件格式：JSONL (.jsonl)
- 文本字段：自动识别 `text` 字段，或 `question`+`answer` 组合

### 输出
- 目录：`output/vocab`
- 文件名：`vocab.pkl.gz`
- 格式：gzip压缩的pickle文件

## 输出示例

```
==================================================
词表构建完成！
构建用时: 123.45 秒
文档总数: 50000
总token数: 15,234,567
词表大小: 234,567
平均文档长度: 304.69
输出文件: output/vocab/vocab.pkl.gz
==================================================
```
