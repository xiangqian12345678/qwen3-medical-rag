# 索引构建模块

医疗知识库索引构建和检索模块，基于 Milvus 实现混合向量检索。

## 功能特性

### 核心功能

1. **多向量字段支持**：支持多个稠密向量字段和稀疏向量字段
   - 稠密向量：chunk、parent_chunk、questions 等
   - 稀疏向量：BM25 基于词表的检索

2. **List 类型字段展开**：对于 `questions` 等列表类型字段，每个元素会展开为单独的向量索引行
   - 支持问题-答案对的精确检索
   - 自动生成唯一的主键（pk）和追踪字段（origin_pk、vector_id）

3. **向量融合检索**：支持 RRF 和加权融合策略
   - RRF（Reciprocal Rank Fusion）：基于排名的融合
   - Weighted Ranker：基于权重的融合

4. **配置驱动**：所有配置通过 `index.yaml` 管理
   - 支持多种嵌入服务：Ollama、DashScope、OpenAI
   - 灵活的索引参数配置

### 技术架构

| 组件 | 功能 | 说明 |
|------|------|------|
| `KnowledgeBase` | 知识库主类 | 提供完整的索引构建和检索功能 |
| `DocumentVectorizer` | 文档向量化器 | 处理多向量字段展开和向量化 |
| `CollectionManager` | Milvus 集合管理器 | 创建 Collection 和索引 |
| `KnowledgeBaseSearcher` | 检索器 | 混合检索和向量融合 |
| `VectorFieldProcessor` | 向量字段处理器 | 稠密向量和稀疏向量的生成 |
| `DataIngestor` | 数据导入器 | 批量导入文档数据 |
| `SparseVectorProcessor` | 稀疏向量处理器 | BM25 稀疏向量生成 |
| `DashScopeEmbeddings` | 千问嵌入客户端 | 阿里云 DashScope 向量服务 |

### 模块结构

```
3.索引构建/
├── config.py                # 配置加载和 Pydantic 模型定义
├── embedding_client.py      # 嵌入模型客户端工厂（Ollama/DashScope/OpenAI）
├── vocab.py                 # 词表管理（稀疏向量）
├── sparse_vectorizer.py     # 稀疏向量处理器（BM25 算法）
├── vectorizer.py            # 文档向量化器（处理多向量字段展开）
├── collection.py            # Milvus Collection 管理器
├── searcher.py              # 检索器（混合检索和向量融合）
├── insert.py                # 数据插入/更新/删除工具
├── knowledge_base.py        # 知识库主类（对外接口）
├── ingest.py                # 数据导入工具
├── run_build_index.py       # 索引构建脚本（命令行工具）
├── run_clean_index.py       # 清理索引脚本
├── run_request_index.py     # 请求索引脚本
├── test_*.py                # 测试脚本
└── index.yaml               # 索引配置文件
```

## 多向量字段展开设计

### 设计背景

对于 `questions` 这类 list 类型的字段，单个文档包含多个问题，每个问题都需要生成独立的向量索引。这需要在 Milvus 中将单个文档展开为多行数据，同时保持与原始文档的关联。

### 解决方案

使用 `origin_pk` 和 `vector_id` 两个字段来追踪原始文档和向量索引的关系：

- **pk**：主键，唯一标识每一行
  - 主行（vector_id=-1）：`pk = origin_pk`
  - 展开行（vector_id>=0）：`pk = {origin_pk}_v{vector_id}`
- **origin_pk**：原始文档的主键（通常是 hash_id），用于追溯同一文档的多条展开记录
- **vector_id**：向量索引序号
  - `-1`：主行，存储 str 类型字段的向量和稀疏向量
  - `>=0`：展开行，存储 list 类型字段中第 vector_id 个元素的向量

### 数据展开示例

**输入文档：**

```python
Document(
    page_content="高血压是一种常见的慢性疾病，需要长期管理...",
    metadata={
        "hash_id": "abc123def456",
        "chunk": "高血压是一种常见的慢性疾病，需要长期管理...",
        "parent_chunk": "高血压管理指南：...",
        "summary": "高血压是一种常见的慢性疾病，需要长期管理。",
        "questions": ["什么是高血压？", "如何管理高血压？", "高血压有哪些症状？"],
        "document": "高血压管理指南",
        "source": "medical_corpus",
        "source_name": "医学手册",
        "lt_doc_id": "doc_001",
        "chunk_id": 0
    }
)
```

**展开后的数据行（3 个问题 + 1 个主行）：**

| pk                | origin_pk | vector_id | chunk | questions                   | questions_dense | chunk_dense   |
|-------------------|-----------|-----------|-------|-----------------------------|-----------------|---------------|
| abc123def456_v0   | abc123... | 0         |       | "什么是高血压？"              | [vec]           | [0.0]*1024    |
| abc123def456_v1   | abc123... | 1         |       | "如何管理高血压？"            | [vec]           | [0.0]*1024    |
| abc123def456_v2   | abc123... | 2         |       | "高血压有哪些症状？"           | [vec]           | [0.0]*1024    |
| abc123def456      | abc123... | -1        | "高血压..." | "什么是高血压？,如何管理..." | [0.0]*1024     | [vec]         |

### 向量填充规则

| 行类型 | chunk_dense | parent_chunk_dense | questions_dense | chunk_sparse |
|--------|-------------|-------------------|------------------|--------------|
| 主行（vector_id=-1） | ✓ 实际向量 | ✓ 实际向量 | 零向量 | ✓ BM25 稀疏向量 |
| 展开行（vector_id>=0） | 零向量 | 零向量 | ✓ 实际向量 | 空字典 |

### 检索处理流程

1. **混合检索**：对多个字段同时进行向量检索
   - `questions_dense`：检索问题相关的语义
   - `chunk_dense`：检索文档内容的语义
   - `parent_chunk_dense`：检索父级文档的上下文

2. **向量融合**：使用 RRF 或加权融合合并多个检索结果
   - **RRF**：基于排名融合，`score = 1 / (k + rank)`
   - **加权融合**：根据配置的权重加权平均

3. **去重**：按 `origin_pk` 去重，保留最相关的结果
   - 同一文档的多个展开行会被合并
   - 保留最高分数的结果

4. **输出**：返回包含完整信息的去重结果

## 使用示例

### 快速构建索引

使用 `run_build_index.py` 脚本从标注数据构建索引：

```bash
# 基本用法：使用默认配置，从 ../output/annotation 加载数据
python run_build_index.py

# 指定配置文件
python run_build_index.py --config index.yaml

# 指定数据目录
python run_build_index.py --data-dir ../output/annotation

# 指定文件模式
python run_build_index.py --file-pattern "*.jsonl"

# 不初始化知识库（增量导入，不删除旧数据）
python run_build_index.py --no-init

# 指定批处理大小
python run_build_index.py --batch-size 100

# 完整参数示例
python run_build_index.py \
  --config index.yaml \
  --data-dir ../output/annotation \
  --file-pattern "*.jsonl" \
  --batch-size 100
```

**命令行参数说明：**

| 参数             | 说明                            | 默认值                      |
|----------------|-------------------------------|---------------------------|
| `--config`      | 配置文件路径                        | `index.yaml`             |
| `--data-dir`    | 数据目录路径（相对或绝对路径）              | `../output/annotation`   |
| `--file-pattern`| 文件匹配模式（支持通配符）                | `*.jsonl`                 |
| `--no-init`     | 不初始化知识库（不删除旧数据，增量导入）         | 否                         |
| `--batch-size`  | 批处理大小（每次插入的文档数）              | 100                       |

### 清理索引

删除 Milvus 中的旧索引数据：

```bash
python run_clean_index.py --config index.yaml
```

### 知识库初始化与文档管理

```python
from knowledge_base import KnowledgeBase
from langchain_core.documents import Document
import hashlib

# 初始化知识库
kb = KnowledgeBase("index.yaml")
kb.initialize()  # 创建 Collection 和索引

# 添加单个文档
chunk = "高血压是一种常见的慢性疾病，需要长期管理。"
hash_id = hashlib.md5(chunk.encode('utf-8')).hexdigest()

doc = Document(
    page_content=chunk,
    metadata={
        "pk": f"{hash_id[:16]}_0",  # 主键，唯一标识
        "chunk": chunk,
        "parent_chunk": "高血压管理指南：...",
        "summary": "高血压是一种常见的慢性疾病，需要长期管理。",
        "questions": ["什么是高血压？", "如何管理高血压？", "高血压有哪些症状？"],
        "document": "高血压管理指南",
        "source": "medical_corpus",
        "source_name": "医学手册",
        "lt_doc_id": "doc_001",
        "chunk_id": 0,
        "hash_id": hash_id  # 用于去重和更新
    }
)
kb.add_documents([doc])

# 插入或更新文档（相同主键会覆盖）
kb.upsert_documents([doc])

# 批量添加文档
documents = [doc1, doc2, doc3, ...]
total_rows = kb.add_documents(documents, show_progress=True)
print(f"插入了 {len(documents)} 个文档，生成 {total_rows} 行数据")
```

### 检索查询

#### 混合检索（推荐）

默认使用所有启用的字段进行混合检索，并自动进行向量融合和去重：

```python
# 默认混合检索
results = kb.search("什么是高血压？", limit=5)

# 自定义检索参数
results = kb.search(
    query="什么是高血压？",
    anns_fields=["questions_dense", "chunk_dense", "parent_chunk_dense"],
    limit=5,
    top_k=50,            # 每个字段检索的 top_k 数量
    fuse=True,           # 启用向量融合
    deduplicate=True     # 按 origin_pk 去重
)

# 输出结果
for result in results:
    entity = result["entity"]
    print(f"Distance: {result['distance']}")
    print(f"Chunk: {entity.get('chunk', '')[:100]}...")
    print(f"Summary: {entity.get('summary', '')[:100]}...")
    print(f"Questions: {entity.get('questions', [])}")
    print(f"Source: {entity.get('source', '')}")
    print("-" * 50)
```

#### 单路检索

仅使用指定字段进行检索，不进行融合：

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

# 只使用 chunk_sparse（BM25）检索
results = kb.simple_search(
    query="高血压 管理",
    anns_field="chunk_sparse",
    limit=5
)
```

### 批量导入数据

```python
from ingest import DataIngestor

# 创建导入器
ingestor = DataIngestor(kb)

# 从文档列表导入
total_rows = ingestor.load_from_documents(
    documents,
    show_progress=True,
    batch_size=100
)

# 从单个文件导入（需要自定义 loader）
documents = my_loader("data.json")
total_rows = ingestor.load_from_documents(documents)

# 从目录导入所有文件
total_files = 0
for file_path in Path("data/").glob("*.txt"):
    documents = my_loader(str(file_path))
    total_rows += ingestor.load_from_documents(documents)
    total_files += 1

print(f"从 {total_files} 个文件导入，共 {total_rows} 行")
```

## 配置说明

### 配置文件结构

`index.yaml` 是索引构建的核心配置文件，包含以下主要部分：

```yaml
milvus:                    # Milvus 连接配置
dense_fields:              # 稠密向量字段配置
sparse_fields:             # 稀疏向量字段配置
base_fields:               # 基础字段配置
fusion:                    # 检索融合配置
default_search:            # 默认检索配置
```

### Milvus 连接配置

```yaml
milvus:
  uri: http://localhost:19530      # Milvus 服务地址
  token: null                      # 认证令牌（Zilliz Cloud 需要）
  collection_name: medical_knowledge  # 集合名称
  drop_old: true                   # 是否删除旧集合（true=重建，false=增量）
  auto_id: false                   # 是否自动生成主键（false=使用 hash_id，推荐）
```

**主键模式说明：**

| 模式 | 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| `auto_id=false`（推荐） | VARCHAR | 支持去重、更新、删除 | 需要手动生成 ID | 生产环境、长期存储 |
| `auto_id=true` | INT64 | 简单，无需管理 ID | 无法去重和更新 | 临时数据、一次性导入 |

### 稠密向量字段配置

```yaml
dense_fields:
  chunk:                    # 字段名称
    embed: true             # 是否向量化
    type: str               # 字段类型：str=字符串，list=列表（会展开）
    dimension: 1024         # 向量维度
    model: bge-m3:latest    # 模型名称
    base_url: http://localhost:11434  # API 地址
    provider: ollama        # 提供商：ollama/dashscope/openai
    workers: 8              # 并发工作线程数
    index_field: chunk_dense  # 索引字段名称
    index_type: HNSW        # 索引类型：HNSW/IVF_FLAT/IVF_PQ
    index_params:           # 索引构建参数
      M: 32                 # HNSW：每个节点的最大连接数
      efConstruction: 200   # HNSW：构建时的搜索宽度
    search_params:          # 检索参数
      ef: 64                # HNSW 检索时的搜索宽度
    metric_type: COSINE     # 距离度量：COSINE/IP/L2
```

**支持的嵌入服务：**

| 提供商 | 模型示例 | 说明 |
|--------|----------|------|
| `ollama` | `bge-m3:latest` | 本地开源模型 |
| `dashscope` | `text-embedding-v2` | 阿里云千问向量服务 |
| `openai` | `text-embedding-3-small` | OpenAI 向量服务 |

**HNSW 索引参数说明：**

- `M`：每个节点的最大连接数（越大索引质量越高，但内存占用越大）
- `efConstruction`：构建时的搜索宽度（越大索引质量越高，但构建越慢）
- `ef`：检索时的搜索宽度（越大检索精度越高，但速度越慢）

### 稀疏向量字段配置（BM25）

```yaml
sparse_fields:
  chunk:
    embed: true                      # 是否启用稀疏向量
    vocab_path: data/vocab.pkl.gz   # 词表文件路径（需预先构建）
    algorithm: BM25                 # 稀疏向量算法
    k1: 1.5                         # BM25：词频饱和度参数
    b: 0.75                         # BM25：文档长度归一化参数
    domain_model: medicine           # 领域模型（用于 pkuseg 分词）
    workers: 8                      # 并发工作线程数
    index_field: chunk_sparse       # 索引字段名称
    index_type: SPARSE_INVERTED_INDEX  # 索引类型
    index_params:
      inverted_index_algo: DAAT_MAXSCORE  # 倒排索引算法
    metric_type: IP                 # 距离度量：IP=内积
```

**BM25 参数说明：**

- `k1`：控制词频的饱和度（通常取 1.2-2.0）
  - 越大，高频词权重越高
- `b`：控制文档长度归一化（通常取 0.75）
  - 越大，长文档受惩罚越大

### 检索融合配置

```yaml
fusion:
  method: rrf                 # 融合方法：rrf 或 weighted
  k: 60                      # RRF 的 k 值（通常取 50-100）
  weights:                   # 加权融合的权重（method=weighted 时使用）
    chunk_dense: 0.35
    parent_chunk_dense: 0.35
    questions_dense: 0.20
    chunk_sparse: 0.10
```

**融合方法对比：**

| 方法 | 公式 | 特点 | 适用场景 |
|------|------|------|----------|
| RRF | `score = Σ 1 / (k + rank)` | 基于排名，无需权重 | 默认推荐 |
| Weighted | `score = Σ weight * normalized_score` | 基于分数，需要配置权重 | 需要精细控制字段贡献 |

### 基础字段配置

```yaml
base_fields:
  - name: pk                 # 字段名称
    datatype: VARCHAR        # 数据类型：VARCHAR/INT64
    max_length: 65535        # VARCHAR 最大长度
    is_primary: true         # 是否为主键
    enable_analyzer: false   # 是否启用分析器（Milvus 全文检索）
```

## API 参考

### KnowledgeBase（知识库主类）

提供完整的知识库功能，是模块的主要对外接口。

#### 初始化

```python
kb = KnowledgeBase(config_path="index.yaml")
```

#### Collection 管理

| 方法 | 说明 |
|------|------|
| `initialize()` | 初始化：创建 Collection 和索引 |
| `create_collection()` | 创建 Collection |
| `build_index()` | 构建索引 |
| `drop_collection()` | 删除 Collection |

#### 文档管理

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `add_documents(docs, show_progress)` | docs: 文档列表<br>show_progress: 是否显示进度条 | int: 插入的行数 | 添加文档（list 字段会展开） |
| `upsert_documents(docs, show_progress)` | docs: 文档列表<br>show_progress: 是否显示进度条 | int: 处理的行数 | 插入或更新文档（相同主键覆盖） |

#### 检索

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `search(query, anns_fields, limit, top_k, output_fields, fuse, deduplicate)` | query: 查询文本<br>anns_fields: 检索字段列表<br>limit: 最终返回数量<br>top_k: 每个字段的 top_k<br>output_fields: 输出字段<br>fuse: 是否融合<br>deduplicate: 是否去重 | List[Dict] | 混合检索 |
| `simple_search(query, anns_field, limit)` | query: 查询文本<br>anns_field: 检索字段<br>limit: 返回数量 | List[Dict] | 单路检索 |

### SearchRequest（检索请求模型）

| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | str | 查询文本 |
| `collection_name` | str | 集合名称 |
| `requests` | List[SingleSearchRequest] | 多路向量查询配置列表 |
| `output_fields` | List[str] | 输出字段列表 |
| `fuse` | FusionSpec | 向量融合策略 |
| `top_k` | int | 融合时每个子请求的返回数量 |
| `limit` | int | 最终返回数量 |

### SingleSearchRequest（单路检索请求模型）

| 参数 | 类型 | 说明 |
|------|------|------|
| `anns_field` | AnnsField | 向量检索字段 |
| `metric_type` | str | 距离计算指标：COSINE/IP/L2 |
| `search_params` | Dict | 检索参数（如 `{"ef": 64}`） |
| `limit` | int | 返回文档数量限制 |
| `expr` | str | 过滤表达式 |

### FusionSpec（向量融合规范）

| 参数 | 类型 | 说明 |
|------|------|------|
| `method` | str | 融合方法：rrf 或 weighted |
| `k` | int | RRF 的 k 值（1-200） |
| `weights` | Dict[str, float] | 加权融合的权重 |

### DataIngestor（数据导入器）

批量导入文档数据。

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `load_from_documents(documents, show_progress, batch_size)` | documents: 文档列表<br>show_progress: 是否显示进度条<br>batch_size: 批次大小 | int: 插入的总行数 | 从文档列表导入 |
| `load_from_file(file_path, loader, show_progress, **kwargs)` | file_path: 文件路径<br>loader: 文档加载函数<br>show_progress: 是否显示进度条<br>**kwargs: 传递给 loader 的参数 | int: 插入的总行数 | 从文件导入 |
| `load_from_directory(directory, loader, show_progress, file_pattern, **kwargs)` | directory: 目录路径<br>loader: 文档加载函数<br>show_progress: 是否显示进度条<br>file_pattern: 文件匹配模式<br>**kwargs: 传递给 loader 的参数 | int: 插入的总行数 | 从目录导入 |

## 测试

### 运行测试

```bash
# 运行完整测试
python test_index.py
```

测试内容包括：

1. 配置加载测试
2. 向量化器测试（多字段展开）
3. 知识库初始化测试
4. 文档添加测试
5. 混合检索和单路检索测试

### 测试用例

```python
from knowledge_base import KnowledgeBase
from langchain_core.documents import Document

# 初始化知识库
kb = KnowledgeBase("index.yaml")
kb.initialize()

# 添加测试文档
doc = Document(
    page_content="测试文档内容",
    metadata={
        "chunk": "测试文档内容",
        "questions": ["测试问题？"],
        "hash_id": "test_hash_001",
        # ... 其他元数据
    }
)
kb.add_documents([doc])

# 执行检索
results = kb.search("测试查询", limit=5)
for result in results:
    print(result)
```

## 依赖

### 核心依赖

| 包名 | 说明 | 用途 |
|------|------|------|
| `pymilvus` | Milvus 客户端 | 向量数据库操作 |
| `langchain-core` | LangChain 核心 | 文档抽象 |
| `langchain-ollama` | Ollama 集成 | 本地向量模型 |
| `langchain-openai` | OpenAI 集成 | OpenAI 向量模型 |
| `pydantic` | 数据验证 | 配置模型定义 |
| `pydantic-settings` | 配置管理 | 环境变量加载 |
| `pyyaml` | YAML 解析 | 配置文件读取 |
| `tqdm` | 进度条 | 批处理进度显示 |
| `pkuseg` | 中文分词 | 稀疏向量分词 |
| `httpx` | HTTP 客户端 | API 调用（DashScope） |

### 可选依赖

| 包名 | 说明 |
|------|------|
| `numpy` | 向量计算（稀疏向量） |

## 最佳实践

### 1. 配置管理

- **首次构建**：`drop_old: true`，`auto_id: false`
- **增量更新**：`drop_old: false`，使用相同的 hash_id
- **索引类型**：小数据集用 IVF_FLAT，大数据集用 HNSW

### 2. 向量嵌入选择

| 场景 | 推荐模型 | 维度 | 提供商 |
|------|----------|------|--------|
| 通用医疗 | bge-m3:latest | 1024 | Ollama |
| 高质量嵌入 | text-embedding-v2 | 1536 | DashScope |
| 快速原型 | text-embedding-3-small | 1536 | OpenAI |

### 3. 融合权重配置

```yaml
fusion:
  weights:
    chunk_dense: 0.40         # 语义匹配（最重要）
    parent_chunk_dense: 0.30  # 上下文匹配
    questions_dense: 0.20      # 问题匹配
    chunk_sparse: 0.10         # 关键词匹配
```

### 4. 性能优化

- **批量大小**：100-200 文档/批次
- **并发数**：workers = min(CPU核心数 - 1, 8)
- **HNSW 参数**：M=16-32，efConstruction=200-400，ef=64-128

## 故障排查

### 常见问题

1. **Milvus 连接失败**
   ```
   错误: Failed to connect to Milvus
   解决: 检查 Milvus 服务是否运行：docker ps | grep milvus
   ```

2. **嵌入模型超时**
   ```
   错误: Timeout waiting for embedding
   解决: 增加 timeout 或减少批处理大小
   ```

3. **词表加载失败**
   ```
   错误: 词表加载失败
   解决: 确保先在 2.词库生成 中构建词表
   ```

4. **向量维度不匹配**
   ```
   错误: dimension mismatch
   解决: 检查 dimension 配置与实际模型输出一致
   ```

## 依赖关系

```
3.索引构建/
├── 依赖 1.数据处理 (标注数据)
├── 依赖 2.词库生成 (稀疏向量词表)
└── 输出到 5.单轮对话RAG (知识库检索)
```

## 更新日志

- **v1.0.0**：初始版本，支持混合向量检索和多向量字段展开
