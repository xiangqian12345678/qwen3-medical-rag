# 文本RAG标准产品

## 概述

文本RAG标准产品是一个基于LangGraph的医疗问答智能体系统，支持多轮对话、主动追问、查询强化、召回增强、文档过滤排序等高级功能。系统集成了向量数据库检索、网络搜索和知识图谱检索三种召回方式，提供完整的RAG解决方案。

## 功能特性

### 核心功能
1. **主动追问**: 根据用户输入判断是否需要追问关键信息
2. **背景信息抽取**: 从多轮对话中提取和整合背景信息
3. **查询强化**: 支持查询重写、多样化查询、上位词查询、假设性答案生成
4. **召回增强**: 支持子查询拆分、多查询并行、上位词生成、假设性答案检索
5. **文档过滤**: 基于LLM或嵌入模型的低相关性过滤、内容过滤、冗余过滤
6. **文档排序**: 支持Cross-Encoder重排序和基于文档位置的重排序
7. **质量判断**: 支持RAG结果质量评估和重试机制

### 召回方式
- **向量数据库检索**: 基于Milvus的稠密/稀疏向量混合检索
- **网络搜索**: 基于LangChain的网络搜索
- **知识图谱检索**: 基于Neo4j的知识图谱检索

## 文件结构说明

```
9.文本RAG标准产品/
├── app_config.py                 # 应用配置类
├── dialogue_agent.py             # 对话Agent主类
├── recall_graph.py               # 召回图（单查询RAG）
├── run_dialogue.py              # 运行入口
├── templates.py                 # 全局提示词模板
├── utils.py                     # 工具函数（LLM/Embedding/Reranker客户端创建）
│
├── answer/                      # 回答生成模块
│   ├── answer.py               # 回答生成逻辑
│   ├── answer_templates.py     # 回答生成提示词模板
│   └── utils.py                # 回答模块工具函数
│
├── enhance/                     # 增强（查询/召回/过滤/排序）模块
│   ├── agent_state.py          # Agent状态定义
│   ├── enhance_templates.py    # 增强相关提示词模板
│   ├── query_enhance.py        # 查询强化（背景抽取、查询重写）
│   ├── recall_enhance.py       # 召回增强（多查询、上位词、假设性答案）
│   ├── filter_enhance.py       # 文档过滤
│   ├── sort_enhance.py         # 文档排序
│   └── utils.py                # 增强模块工具函数
│
├── rag/                         # RAG配置模块
│   ├── rag_config.py           # RAG配置数据模型
│   ├── rag_config.yaml         # RAG主配置文件
│   └── rag_loader.py           # RAG配置加载器
│
├── recall/                      # 召回模块
│   ├── milvus/                 # 向量数据库召回
│   │   ├── embed_config.py     # Milvus配置数据模型
│   │   ├── embed_config.yaml   # Milvus配置文件
│   │   ├── embed_loader.py     # Milvus配置加载器
│   │   ├── embed_search.py     # 向量检索逻辑
│   │   ├── embed_searcher.py   # 向量检索器
│   │   ├── embed_templates.py  # 向量检索提示词模板
│   │   ├── embed_utils.py      # 向量检索工具函数
│   │   ├── embed_vocab.py      # 稀疏向量词表
│   │   └── sparse_vectorizer.py  # 稀疏向量化器
│   │
│   ├── search/                 # 网络召回
│   │   ├── search_config.py    # 网络搜索配置数据模型
│   │   ├── search_config.yaml  # 网络搜索配置文件
│   │   ├── search_loader.py    # 网络搜索配置加载器
│   │   ├── web_search.py       # 网络搜索逻辑
│   │   ├── web_searcher.py     # 网络搜索器
│   │   ├── search_templates.py # 网络搜索提示词模板
│   │   └── search_utils.py     # 网络搜索工具函数
│   │
│   └── kgraph/                 # 知识图谱召回
│       ├── kg_config.py       # 知识图谱配置数据模型
│       ├── kg_config.yaml     # 知识图谱配置文件
│       ├── kg_loader.py       # 知识图谱配置加载器
│       ├── kgraph_schema.py   # 知识图谱Schema定义
│       ├── kgraph_search.py   # 知识图谱检索逻辑
│       ├── kgraph_searcher.py # 知识图谱检索器
│       ├── kg_templates.py    # 知识图谱检索提示词模板
│       ├── kg_utils.py        # 知识图谱工具函数
│       └── neo4j_connection.py # Neo4j连接管理
│
└── test/                       # 测试模块
    ├── test_milvus.py         # Milvus检索测试
    ├── test_search.py         # 网络搜索测试
    ├── test_kgraph.py         # 知识图谱检索测试
    └── testOllamaReranker.py  # Ollama Reranker测试
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `app_config.py` | 应用配置类，整合所有配置 |
| `dialogue_agent.py` | 对话Agent主类，管理多轮对话流程 |
| `recall_graph.py` | 召回图，执行单查询的RAG检索 |
| `run_dialogue.py` | 运行入口，交互式对话 |
| `utils.py` | 工具函数，创建LLM/Embedding/Reranker客户端 |
| `answer/` | 回答生成模块 |
| `enhance/` | 增强模块（查询/召回/过滤/排序） |
| `rag/` | RAG配置管理模块 |
| `recall/` | 召回模块（向量/网络/知识图谱） |
| `test/` | 测试模块 |

## 配置文件说明

### 1. 主配置文件 - rag/rag_config.yaml

#### LLM配置
```yaml
llm:
  provider: dashscope           # 提供商: dashscope/ollama/openai
  model: qwen-plus              # 模型名称
  api_key: sk-xxx               # API密钥
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  temperature: 0.3              # 温度参数
  max_tokens: 2048              # 最大生成token数
```

#### Embedding配置
```yaml
embedding:
  provider: dashscope           # 提供商: dashscope/ollama/openai
  model: text-embedding-v2      # 模型名称
  api_key: sk-xxx               # API密钥
  base_url: https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
  dimension: 1536               # 向量维度
```

#### Reranker配置
```yaml
reranker:
  provider: dashscope           # 提供商: dashscope/ollama
  model: text-reranker-v2       # 模型名称
  api_key: sk-xxx               # API密钥
  base_url: https://dashscope.aliyuncs.com/api/v1/services/rerankers/text-reranker/text-reranker
```

#### Agent配置
```yaml
agent:
  # 对话配置
  max_attempts: 2                # RAG最大重试次数
  console_debug: false           # 控制台调试开关
  max_ask_num: 5                 # 最大追问轮次

  # 网络搜索配置
  network_search_enabled: true   # 是否启用网络搜索
  network_search_cnt: 10         # 网络搜索结果数量

  # 知识图谱搜索配置
  kgraph_search_enabled: true    # 是否启用知识图谱搜索
  kgraph_search_cnt: 10          # 知识图谱搜索结果数量

  # Query强化
  query_intent_enabled: false         # 意图识别（预留）
  query_rewrite_enabled: true        # 查询重写
  query_refine_enabled: true         # 背景信息抽取

  # 召回增强
  generate_sub_queries_enabled: false                 # 生成子问题
  generate_multi_queries_enabled: false               # 生成多个问题
  generate_superordinate_query_enabled: true          # 生成上位词
  generate_hypothetical_answer_enabled: true          # 生成假设性答案

  # 过滤
  filter_low_correction_content_enabled: true        # 过滤低相关内容
  filter_low_correction_doc_llm_enabled: false        # 基于大模型过滤低相关文档
  filter_low_correction_doc_embeddings_enabled: true  # 基于嵌入模型过滤低相关文档
  low_correction_threshold: 0.65                     # 低相关性阈值
  filter_redundant_doc_embeddings_enabled: false    # 基于向量引擎过滤冗余文档
  redundant_threshold: 0.95                           # 冗余阈值

  # 排序
  sort_docs_cross_encoder_enabled: true           # 基于Cross-Encoder重排序
  sort_docs_by_loss_of_location_enabled: true      # 基于长文本的重排序
```

#### Dialogue配置
```yaml
dialogue:
  estimate_token_fun: avg        # Token估算函数: avg/claude
  llm_max_token: 1024            # LLM最大token数
  max_token_threshold: 1.01      # 最大token阈值
  cut_dialogue_scale: 2          # 对话截断倍数
  smith_debug: false             # Smith调试开关
  console_debug: true             # 控制台调试开关
  thinking_in_context: false     # 上下文思考
  cache_time: 60                 # 会话历史和摘要缓存超时时间（分钟）
  summary_max_length: 500         # 每次摘要生成的最大字符长度
  summary_max_cache_count: 3      # 缓存的摘要迭代次数上限
```

### 2. Milvus配置 - recall/milvus/embed_config.yaml

```yaml
# Milvus配置
milvus:
  uri: http://localhost:19530     # Milvus服务地址
  token: null                     # 认证令牌
  collection_name: medical_knowledge  # 集合名称
  drop_old: false                 # 是否删除旧集合
  auto_id: false                  # 是否自动生成ID

# 稠密向量字段配置
dense_fields:
  chunk:                          # chunk字段配置
    embed: true                   # 是否创建向量
    index_field: chunk_dense      # 索引字段名
    index_type: HNSW              # 索引类型
    search_params:
      ef: 64                      # 搜索参数
    metric_type: COSINE           # 距离度量

# 稀疏向量字段配置
sparse_fields:
  chunk:                          # chunk字段配置
    embed: true
    vocab_path: ../output/vocab/vocab.pkl.gz  # 词表路径
    algorithm: BM25               # BM25算法
    k1: 1.5                       # BM25参数k1
    b: 0.75                       # BM25参数b

# 检索融合配置
fusion:
  method: rrf                     # 融合方法: rrf/weighted_sum
  k: 60                           # RRF参数k
  weights:                        # 各字段权重
    chunk_dense: 0.35
    parent_chunk_dense: 0.35
    questions_dense: 0.20
    chunk_sparse: 0.10
```

### 3. 知识图谱配置 - recall/kgraph/kg_config.yaml

```yaml
neo4j:
  uri: bolt://localhost:7687      # Neo4j服务地址
  user: neo4j                     # 用户名
  password: "12345678"           # 密码
  database: neo4j                 # 数据库名称
  max_connection_lifetime: 3600  # 最大连接生命周期
  max_connection_pool_size: 50    # 最大连接池大小
  connection_timeout: 30.0         # 连接超时时间
```

## 提示词模板说明

### 1. 回答生成模板 (answer/answer_templates.py)

| 模板名称 | 说明 |
|---------|------|
| `basic_rag` | 基础RAG问答模板，基于文档回答用户问题 |
| `judge_rag` | RAG结果评判模板，判断回答是否遵循事实 |
| `ask_user` | 主动追问模板，判断是否需要向用户追问关键信息 |
| `extract_user_info` | 用户信息提取模板，从对话中抽取背景信息 |

### 2. 增强相关模板 (enhance/enhance_templates.py)

| 模板名称 | 说明 |
|---------|------|
| `rewrite_query` | 查询重写模板，将口语化问题改写为专业检索词 |
| `multi_query` | 多样化查询模板，生成5个不同版本的改写问题 |
| `superordinate_query` | 上位词查询模板，生成更抽象的上位问题 |
| `hypothetical_answer` | 假设性回答模板，生成假设性答案用于检索 |

### 3. 向量检索模板 (recall/milvus/embed_templates.py)

| 模板名称 | 说明 |
|---------|------|
| `call_db` | 向量数据库检索调用模板 |

### 4. 网络搜索模板 (recall/search/search_templates.py)

| 模板名称 | 说明 |
|---------|------|
| `call_network` | 网络搜索调用模板 |

### 5. 知识图谱检索模板 (recall/kgraph/kg_templates.py)

| 模板名称 | 说明 |
|---------|------|
| `call_kgraph` | 知识图谱检索调用模板 |

## 参数配置代码说明

### 配置数据模型 (rag/rag_config.py)

```python
class LLMConfig(BaseModel):
    """LLM配置"""
    provider: Literal['openai', 'dashscope', 'ollama']
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    temperature: float
    max_tokens: Optional[int]

class EmbeddingConfig(BaseModel):
    """Embedding配置"""
    provider: Literal['openai', 'dashscope', 'ollama']
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    dimension: Optional[int]

class RerankerConfig(BaseModel):
    """Reranker配置"""
    provider: Literal['dashscope', 'ollama']
    model: str
    base_url: Optional[str]
    api_key: Optional[str]

class AgentConfig(BaseModel):
    """Agent对话配置"""
    max_attempts: int
    console_debug: bool
    max_ask_num: int
    network_search_enabled: bool
    network_search_cnt: int
    kgraph_search_enabled: bool
    kgraph_search_cnt: int
    # ... 其他配置项

class DialogueConfig(BaseModel):
    """多轮对话RAG配置"""
    estimate_token_fun: str
    llm_max_token: int
    max_token_threshold: float
    cut_dialogue_scale: int
    # ... 其他配置项
```

### 配置加载器 (rag/rag_loader.py)

```python
from rag.rag_config import RAGConfig

class RAGConfigLoader:
    """RAG配置加载器"""

    def __init__(self, config_path: str = "rag/rag_config.yaml"):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        self.config = RAGConfig(**config_dict)
```

## 代码架构

### 多轮对话Agent状态流转

```
用户输入
    ↓
[ask_user] 判断是否需要追问 ──────────────┐
    ↓                               │
需要追问? → 是 → [END]              │
    ↓ 否                              │
[query_enhance] 查询强化             │
    ├─ 背景信息抽取                  │
    └─ 查询重写                      │
    ↓                               │
[recall_enhance] 召回增强           │
    ├─ 生成多个问题（可选）          │
    ├─ 生成上位词（可选）            │
    ├─ 生成假设性答案（可选）        │
    └─ 拆分子问题（可选）            │
    ↓                               │
[recall] 并行执行召回                │
    ├─ 向量数据库召回                 │
    ├─ 网络召回（可选）               │
    └─ 知识图谱召回（可选）           │
    ↓                               │
[filter_enhance] 文档过滤           │
    ├─ 低相关性文档过滤              │
    ├─ 内容过滤                      │
    └─ 冗余文档过滤（可选）           │
    ↓                               │
[sort_enhance] 文档排序             │
    ├─ Cross-Encoder重排序          │
    ├─ 去重                          │
    └─ 基于文档位置的重排序           │
    ↓                               │
[answer] 生成答案并写入对话历史       │
    ↓                               │
返回结果 ───────────────────────────┘
```

### RecallGraph状态流转（单查询RAG）

```
查询输入
    ↓
[db_search] 向量数据库检索
    ↓
[web_search] 网络检索 (可选)
    ↓
[kgraph_search] 知识图谱检索 (可选)
    ↓
[rag] RAG生成回答
    ↓
[judge] 质量判断
    ├─ 通过 → [finish_success]
    ├─ 失败 → [finish_fail]
    └─ 重试 → [rag]
```

## 如何使用

### 基本使用

```python
from app_config import APPConfig
from recall.kgraph.kg_loader import KGraphConfigLoader
from recall.milvus.embed_loader import EmbedConfigLoader
from dialogue_agent import DialogueAgent
from rag.rag_loader import RAGConfigLoader
from utils import create_llm_client, create_embedding_client, create_reranker_client

# 1. 初始化配置
rag_config_loader = RAGConfigLoader()
milvus_config_loader = EmbedConfigLoader()
kgraph_config_loader = KGraphConfigLoader()
app_config = APPConfig(rag_config_loader, milvus_config_loader, kgraph_config_loader)

# 2. 创建LLM/Embedding/Reranker客户端
llm_model = create_llm_client(rag_config_loader.config.llm)
embed_model = create_embedding_client(rag_config_loader.config.embedding)
reranker = create_reranker_client(rag_config_loader.config.reranker)

# 3. 初始化Agent
agent = DialogueAgent(app_config=app_config, embeddings_model=embed_model, llm=llm_model, reranker=reranker)

# 4. 调用Agent
state = agent.answer(query="什么是房颤？")

# 5. 获取结果
print(state["final_answer"])
```

### 交互式对话

```bash
python run_dialogue.py
```

### 配置选项切换

通过修改 `rag/rag_config.yaml` 文件中的配置项来启用/禁用功能：

```yaml
# 启用网络搜索
agent:
  network_search_enabled: true

# 启用知识图谱搜索
agent:
  kgraph_search_enabled: true

# 启用查询重写
agent:
  query_rewrite_enabled: true

# 启用上位词生成
agent:
  generate_superordinate_query_enabled: true

# 启用假设性答案生成
agent:
  generate_hypothetical_answer_enabled: true

# 启用Cross-Encoder重排序
agent:
  sort_docs_cross_encoder_enabled: true
```

## 如何测试

### 1. 向量数据库检索测试

```bash
python test_milvus.py
```

测试内容：
- 直接调用检索工具
- LLM数据库检索节点
- LLM判断数据库检索
- 批量查询
- 基于已有文档的检索

### 2. 网络搜索测试

```bash
python test_search.py
```

测试内容：
- 直接网络搜索
- LLM网络搜索节点
- LLM判断无需搜索
- 批量网络搜索
- 完整工作流测试

### 3. 知识图谱检索测试

```bash
python test_kgraph.py
```

测试内容：
- 关键词检索
- 关系检索
- 综合图谱检索
- 检索工具调用
- LLM判断图谱检索

## Milvus索引字段信息

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `chunk` | VARCHAR | 文本块内容 |
| `parent_chunk` | VARCHAR | 父文本块内容 |
| `summary` | VARCHAR | 摘要内容 |
| `questions` | VARCHAR | 问题内容 |
| `source` | VARCHAR | 数据来源 |
| `source_name` | VARCHAR | 数据来源名称 |
| `document` | VARCHAR | 文档内容 |
| `origin_pk` | VARCHAR | 文档唯一标识 |
| `chunk_dense` | FLOAT_VECTOR | 文本块稠密向量 |
| `parent_chunk_dense` | FLOAT_VECTOR | 父文本块稠密向量 |
| `questions_dense` | FLOAT_VECTOR | 问题稠密向量 |
| `chunk_sparse` | SPARSE_FLOAT_VECTOR | 文本块稀疏向量(BM25) |

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

修改 `rag/rag_config.yaml` 文件中的 `api_key` 字段：

```yaml
llm:
  api_key: sk-your_api_key_here

embedding:
  api_key: sk-your_api_key_here

reranker:
  api_key: sk-your_api_key_here
```

获取 API Key: https://dashscope.console.aliyun.com/apiKey

### 3. 启动外部服务

**Milvus服务**
```bash
# 确保Milvus在 http://localhost:19530 运行
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 \
  -v /path/to/milvus:/var/lib/milvus \
  milvusdb/milvus:latest
```

**Neo4j服务**
```bash
# 确保Neo4j在 bolt://localhost:7687 运行
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/12345678 \
  neo4j:latest
```

## 依赖说明

- LangChain: LLM框架
- LangGraph: 状态图框架
- PyMilvus: 向量数据库客户端
- Neo4j: 图数据库客户端
- Pydantic: 数据校验
- DashScope: 阿里云大模型API
- Ollama: 本地大模型服务
