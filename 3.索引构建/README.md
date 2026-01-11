# 索引构建模块

医疗知识库索引构建和检索模块，基于 Milvus 实现混合向量检索。

## 架构设计

### 核心特性

1. **多向量字段支持**：支持多个稠密向量字段和稀疏向量字段
2. **List 类型字段展开**：对于 `questions` 等列表类型字段，每个元素会展开为单独的向量索引行
3. **向量融合检索**：支持 RRF 和加权融合策略
4. **配置驱动**：所有配置通过 `index.yaml` 管理

### 模块结构

```
3.索引构建/
├── config.py                # 配置加载和模型定义
├── embedding_client.py      # 嵌入模型客户端工厂
├── vocab.py                 # 词表管理（稀疏向量）
├── sparse_vectorizer.py     # 稀疏向量处理器（BM25）
├── vectorizer.py            # 文档向量化器（处理多向量字段展开）
├── collection.py            # Milvus Collection 管理器
├── searcher.py              # 检索器（混合检索和向量融合）
├── insert.py                # 数据插入工具
├── knowledge_base.py        # 知识库主类
├── ingest.py                # 数据导入工具
├── index.py                 # 索引构建脚本
├── test_index.py            # 测试脚本
└── index.yaml               # 索引配置文件
```

## 多向量字段展开设计

### 问题

对于 `questions` 这类 list 类型的字段，单个文档包含多个问题，每个问题都需要生成独立的向量索引。

### 解决方案

使用 `origin_pk` 和 `vector_id` 两个字段来追踪原始文档和向量索引的关系：

- `origin_pk`: 原始文档的主键（如 hash_id）
- `vector_id`: 向量索引的序号（-1 表示主行，>=0 表示展开的向量行）

### 数据展开示例

输入文档：

```python
Document(
    page_content="高血压是一种常见的慢性疾病...",
    metadata={
        "hash_id": "abc123",
        "chunk": "高血压...",
        "questions": ["什么是高血压？", "如何管理高血压？"],
        "summary": "高血压摘要..."
    }
)
```

展开后的数据行：

| pk        | origin_pk | vector_id | chunk    | questions  | questions_dense | chunk_dense | summary_dense |
|-----------|-----------|-----------|----------|------------|-----------------|-------------|---------------|
| abc123_v0 | abc123    | 0         |          | "什么是高血压？"  | [vec]           | []          | []            |
| abc123_v1 | abc123    | 1         |          | "如何管理高血压？" | [vec]           | []          | []            |
| abc123    | abc123    | -1        | "高血压..." |            | []              | [vec]       | [vec]         |

### 检索处理

1. **混合检索**：对所有字段进行检索，然后使用 RRF 或加权融合
2. **去重**：按 `origin_pk` 去重，保留最相关的结果
3. **输出**：返回包含完整信息的去重结果

## 使用示例

### 索引构建

使用 `index.py` 脚本从标注数据构建索引：

```bash
# 基本用法：使用默认配置
python index.py

# 指定配置文件
python index.py --config ../config/index.yaml

# 指定数据目录
python index.py --data-dir ../output/annotation

# 指定文件模式
python index.py --file-pattern "*.jsonl"

# 不初始化知识库（增量导入，不删除旧数据）
python index.py --no-init

# 指定批处理大小
python index.py --batch-size 50

# 完整参数示例
python index.py \
  --config ../config/index.yaml \
  --data-dir ../output/annotation \
  --file-pattern "*.jsonl" \
  --batch-size 100
```

**命令行参数说明：**

| 参数               | 说明              | 默认值                    |
|------------------|-----------------|------------------------|
| `--config`       | 配置文件路径          | `index.yaml`           |
| `--data-dir`     | 数据目录路径          | `../output/annotation` |
| `--file-pattern` | 文件匹配模式          | `*.jsonl`              |
| `--no-init`      | 不初始化知识库（不删除旧数据） | 否                      |
| `--batch-size`   | 批处理大小           | 100                    |

### 基本使用

```python
from knowledge_base import MedicalKnowledgeBase
from langchain_core.documents import Document

# 初始化知识库
kb = MedicalKnowledgeBase("index.yaml")
kb.initialize()  # 创建 Collection 和索引

# 添加文档
doc = Document(
    page_content="高血压是一种常见的慢性疾病，需要长期管理。",
    metadata={
        "chunk": "高血压是一种常见的慢性疾病，需要长期管理。",
        "parent_chunk": "高血压管理指南：...",
        "summary": "高血压是一种常见的慢性疾病，需要长期管理。",
        "questions": ["什么是高血压？", "如何管理高血压？"],
        "document": "高血压管理指南",
        "source": "医学手册",
        "source_name": "manual",
        "lt_doc_id": "doc_001",
        "chunk_id": 0,
        "hash_id": "abc123"
    }
)
kb.add_documents([doc])

# 检索
results = kb.search("什么是高血压？", limit=5)
for result in results:
    entity = result["entity"]
    print(f"Score: {result['distance']}")
    print(f"Chunk: {entity['chunk']}")
    print(f"Summary: {entity['summary']}")
```

### 单路检索

```python
# 只使用 questions 字段检索
results = kb.simple_search(
    query="什么是高血压？",
    anns_field="questions_dense",
    limit=5
)

# 只使用 chunk 字段检索
results = kb.simple_search(
    query="高血压管理",
    anns_field="chunk_dense",
    limit=5
)
```

### 混合检索（指定字段）

```python
# 指定检索字段和融合方式
results = kb.search(
    query="什么是高血压？",
    anns_fields=["questions_dense", "chunk_dense", "summary_dense"],
    limit=5,
    top_k=50,
    fuse=True,  # 启用融合
    deduplicate=True  # 按 origin_pk 去重
)
```

### 批量导入

```python
from ingest import DataIngestor

ingestor = DataIngestor(kb)

# 从文档列表导入
ingestor.load_from_documents(documents, batch_size=100)

# 从文件导入
ingestor.load_from_file("data.json", my_loader)

# 从目录导入
ingestor.load_from_directory("data/", my_loader, file_pattern="*.txt")
```

## 配置说明

### 稠密向量字段配置

```yaml
dense_fields:
  questions:
    embed: true              # 是否向量化
    type: list              # 字段类型：str 或 list
    dimension: 1024         # 向量维度
    model: bge-m3:latest    # 模型名称
    base_url: http://localhost:11434
    provider: ollama
    workers: 8
    index_field: questions_dense  # 索引字段名称
    index_type: HNSW
    index_params:
      M: 32
      efConstruction: 200
    search_params:
      ef: 64
    metric_type: COSINE
```

### 稀疏向量字段配置

```yaml
sparse_fields:
  chunk:
    embed: true
    vocab_path: data/vocab.pkl.gz
    algorithm: BM25
    k1: 1.5
    b: 0.75
    domain_model: medicine
    workers: 8
    index_field: chunk_sparse
    index_type: SPARSE_INVERTED_INDEX
    index_params:
      inverted_index_algo: DAAT_MAXSCORE
    metric_type: IP
```

### 检索融合配置

```yaml
fusion:
  method: rrf  # 融合方法：rrf 或 weighted
  k: 60       # RRF 的 k 值
  weights: # 加权融合的权重
    chunk_dense: 0.25
    parent_chunk_dense: 0.25
    summary_dense: 0.25
    questions_dense: 0.15
    chunk_sparse: 0.1
```

## API 参考

### MedicalKnowledgeBase

主类，提供完整的知识库功能。

#### 方法

| 方法                                        | 说明                    |
|-------------------------------------------|-----------------------|
| `initialize()`                            | 初始化：创建 Collection 和索引 |
| `create_collection()`                     | 创建 Collection         |
| `build_index()`                           | 构建索引                  |
| `add_documents(docs, show_progress)`      | 添加文档                  |
| `upsert_documents(docs, show_progress)`   | 插入或更新文档               |
| `search(query, ...)`                      | 混合检索                  |
| `simple_search(query, anns_field, limit)` | 单路检索                  |

### SearchRequest

检索请求模型。

| 参数                | 说明         |
|-------------------|------------|
| `query`           | 查询文本       |
| `collection_name` | 集合名称       |
| `requests`        | 多路向量查询配置列表 |
| `output_fields`   | 输出字段列表     |
| `fuse`            | 向量融合策略     |
| `limit`           | 最终返回数量     |

## 索引构建测试

```bash
# 构建完整索引
python index.py
```

## 测试

```bash
python test_index.py
```

测试内容包括：

1. 配置加载
2. 向量化器（多字段展开）
3. 知识库初始化
4. 文档添加
5. 混合检索和单路检索

## 依赖

- `pymilvus`: Milvus 客户端
- `langchain-core`: 文档抽象
- `langchain-openai` / `langchain-ollama`: 嵌入模型
- `pydantic`: 配置模型验证
- `pydantic-settings`: 配置管理
- `pyyaml`: YAML 配置解析
- `tqdm`: 进度条
- `pkuseg`: 中文分词
- `numpy`: 向量计算

## 注意事项

1. 确保 Milvus 服务正在运行
2. 确保 Ollama 或其他嵌入模型服务可用
3. 稀疏向量需要预先构建词表
4. List 类型字段展开会增加数据量，注意存储和检索性能
