# 多轮对话RAG

多轮对话医疗RAG系统，支持会话管理、历史压缩、查询改写、混合检索等功能。

---

## 一、使用说明

### 1.1 环境准备

确保以下服务已启动：

- **Milvus** 向量数据库（默认 `http://localhost:19530`）
- **Ollama** 服务（默认 `http://localhost:11434`）
- 需要已拉取的模型：
  - `bge-m3:latest` - 嵌入模型
  - `qwen3:4b` - 对话模型

### 1.2 配置文件

编辑 `multi_dialogue.yaml` 配置文件：

```yaml
# Milvus 配置
milvus:
  uri: http://localhost:19530
  collection_name: medical_knowledge

# 稠密向量配置
dense_fields:
  chunk:
    provider: ollama
    model: bge-m3:latest
    base_url: http://localhost:11434
    dimension: 1024

# 稀疏向量配置
sparse_fields:
  chunk:
    provider: self
    vocab_path_or_name: ../output/vocab/vocab.pkl.gz
    algorithm: BM25
    k1: 1.5
    b: 0.75

# LLM 配置
llm:
  provider: ollama
  model: qwen3:4b
  base_url: http://localhost:11434
  temperature: 0.1

# 多轮对话配置
multi_dialogue_rag:
  estimate_token_fun: avg
  llm_max_token: 1024
  cut_dialogue_scale: 2
  max_token_threshold: 1.01
```

### 1.3 运行示例

```bash
cd 6.多轮对话RAG
python run_example.py
```

### 1.4 代码集成

```python
from config import ConfigLoader
from multi_dialogue_rag import MultiDialogueRag

# 加载配置
config_loader = ConfigLoader()

# 创建多轮对话RAG系统
rag = MultiDialogueRag(config_loader.config)

# 提问（指定session_id维护会话上下文）
result = rag.generate_answer(
    query="什么是高血压？",
    session_id="user_001",
    return_document=True
)

print(result["answer"])
```

---

## 二、代码结构

```
6.多轮对话RAG/
├── __init__.py
├── multi_dialogue.yaml          # 配置文件
├── README.md                    # 说明文档
│
├── multi_dialogue_rag.py         # 多轮对话RAG核心实现
├── rag_base.py                   # RAG抽象基类
│
├── knowledge_base.py             # 知识库封装（Milvus操作）
├── retriever.py                 # LangChain标准检索器
│
├── prompts.py                   # Prompt模板管理
├── utils.py                     # 工具函数（LLM、Embedding创建）
│
├── config/                      # 配置模块
│   ├── __init__.py
│   ├── loader.py               # 配置加载器
│   └── models.py               # 配置数据模型
│
├── embed/                       # 嵌入模块
│   ├── __init__.py
│   ├── vocab.py                # 词表加载
│   └── bm25.py                 # BM25稀疏向量
│
└── run_example.py               # 运行示例
```

### 2.1 核心模块说明

| 文件 | 说明 |
|------|------|
| `multi_dialogue_rag.py` | 多轮对话RAG主类，处理对话历史管理、查询改写、文档上下文构建 |
| `knowledge_base.py` | 封装Milvus操作，支持单路检索和混合检索（RRF/Weighted） |
| `retriever.py` | LangChain标准检索器接口实现 |
| `prompts.py` | 管理所有Prompt模板（对话RAG、查询改写、摘要生成） |
| `utils.py` | 创建LLM/Embedding客户端、Token估算函数注册 |
| `rag_base.py` | RAG抽象基类，定义统一接口 |

### 2.2 配置数据模型 (`config/models.py`)

- `AppConfig` - 应用主配置
- `MilvusConfig` - Milvus连接配置
- `DenseFieldConfig` - 稠密向量配置
- `SparseFieldConfig` - 稀疏向量配置（BM25）
- `LLMConfig` - LLM配置
- `MultiDialogueRagConfig` - 多轮对话配置
- `SearchRequest` - 混合检索请求

---

## 三、流程图

### 3.1 整体处理流程

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────────┐
│  检查对话历史是否过长                             │
│  ↓                                               │
│  若超限 → 压缩旧对话为摘要 → 保留最新对话        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  查询改写                                        │
│  输入：用户问题 + 对话历史                        │
│  输出：独立完整的检索查询                         │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  混合检索（Milvus）                              │
│  ├─ 稠密检索（chunk_dense）                      │
│  ├─ 稠密检索（parent_chunk_dense）               │
│  ├─ 稠密检索（questions_dense）                  │
│  ├─ 稀疏检索（chunk_sparse BM25）                │
│  └─ 结果融合（RRF/Weighted）                     │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  动态文档上下文构建                               │
│  ┌──────────────────────────────────────────┐  │
│  │ 计算剩余token预算                         │  │
│  │   总token预算 = llm_max_token             │  │
│  │   已用 = 系统prompt + 历史对话 + 用户提问  │  │
│  │   剩余 = 总预算 - 已用                     │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │ 根据剩余token截取文档内容                  │  │
│  │   - 按优先级依次添加文档                   │  │
│  │   - 剩余不足时截断当前文档                 │  │
│  │   - 后续文档不再添加                       │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  答案生成                                        │
│  输入：系统prompt + 运行摘要 + 对话历史 + 文档上下文 + 用户问题 │
│  输出：AI回答                                     │
└─────────────────────────────────────────────────┘
    │
    ▼
  返回答案
```

### 3.2 会话管理流程

```
新会话
    │
    ▼
┌─────────────────────────────────────────────────┐
│  初始化会话                                      │
│  - 创建 ChatMessageHistory                       │
│  - 初始化 running_summary = ""                   │
│  - 初始化 token元数据                            │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  处理用户问题                                    │
│  ┌──────────────────────────────────────────┐  │
│  │ 1. 检查是否需要压缩历史                   │  │
│  │    若需要：生成摘要 → 清除旧消息          │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │ 2. 查询改写 → 检索 → 生成答案            │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │ 3. 更新对话历史                           │  │
│  │    添加用户消息 + AI回答                  │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
    │
    ▼
  保存会话状态（session_id → history, summary）
```

### 3.3 历史压缩流程

```
对话历史
    │
    ▼
┌─────────────────────────────────────────────────┐
│  估算token数                                    │
│  方法选择：                                      │
│  - avg：基于字符数平均token率估算                │
│  - tiktoken：使用tiktoken库精确估算             │
└─────────────────────────────────────────────────┘
    │
    ▼ (超限?)
┌─────────────────────────────────────────────────┐
│  确定需要保留的最新消息数                        │
│  keep_count = len(messages) // cut_dialogue_scale │
│  例如：8条消息，cut_dialogue_scale=2 → 保留4条   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  提取旧消息生成摘要                              │
│  old_messages = messages[:n]                     │
│  new_messages = messages[n:]                     │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  调用LLM生成摘要                                 │
│  Prompt：保留核心医疗信息 + 用户意图 + 自然口吻  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  合并摘要                                        │
│  running_summary = prev_summary + new_summary   │
└─────────────────────────────────────────────────┘
    │
    ▼
  截断对话历史，保留最新消息
```

### 3.4 混合检索流程

```
改写后的查询
    │
    ▼
┌─────────────────────────────────────────────────┐
│  编码查询向量                                    │
│  ├─ chunk_dense: bge-m3 嵌入                    │
│  ├─ parent_chunk_dense: bge-m3 嵌入             │
│  ├─ questions_dense: bge-m3 嵌入                │
│  └─ chunk_sparse: BM25 稀疏向量                 │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  执行多路检索                                    │
│  每路检索返回 top_k 结果                         │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  结果融合                                        │
│  ┌──────────────────────────────────────────┐  │
│  │ RRF (Reciprocal Rank Fusion)              │  │
│  │   score = Σ 1 / (k + rank)                │  │
│  └──────────────────────────────────────────┘  │
│  或                                             │
│  ┌──────────────────────────────────────────┐  │
│  │ Weighted (加权融合)                       │  │
│  │   score = Σ weight_i * score_i            │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
    │
    ▼
  返回融合后的 top_n 结果
```

---

## 四、主要特性

| 特性 | 说明 |
|------|------|
| **会话隔离** | 通过 `session_id` 区分不同用户会话，每个会话独立管理历史 |
| **历史压缩** | 自动检测对话长度，生成摘要释放token空间 |
| **查询改写** | 基于对话历史将当前问题改写为独立查询 |
| **混合检索** | 支持多路稠密/稀疏检索 + RRF/Weighted融合 |
| **动态上下文** | 根据token预算动态调整文档上下文内容 |
| **Token估算** | 支持多种token估算方法（avg/tiktoken） |
| **LangChain集成** | 使用LangChain标准接口，易于扩展 |

---

## 五、配置参数说明

### 5.1 多轮对话配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `estimate_token_fun` | Token估算方法 | `avg` |
| `llm_max_token` | LLM最大token数 | `1024` |
| `cut_dialogue_scale` | 历史压缩比例（越大保留越多） | `2` |
| `max_token_threshold` | 最大token阈值倍数 | `1.01` |
| `console_debug` | 控制台调试输出 | `false` |
| `thinking_in_context` | 上下文思考模式 | `false` |

### 5.2 检索配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `fuse.method` | 融合方法：`rrf`/`weighted` | `weighted` |
| `fuse.weights` | 加权融合的权重列表 | `[0.8, 0.2]` |
| `fuse.k` | RRF融合的k参数 | `60` |
| `top_k` | 每路检索返回数 | `50` |
| `limit` | 最终返回结果数 | `5` |

---

## 六、扩展自定义

### 6.1 自定义Token估算函数

```python
from enhance.utils import register_estimate_function


@register_estimate_function("my_method")
def my_estimate_function(text: str) -> int:
    # 自定义估算逻辑
    return len(text) // 4
```

然后在配置中设置 `estimate_token_fun: my_method`

### 6.2 自定义Prompt模板

```python
from prompts import register_prompt_template

register_prompt_template("my_template", {
    "system": "你是一个...",
    "user": "..."
})
```

### 6.3 自定义检索配置

```python
from config.models import SingleSearchRequest, FusionSpec

# 创建自定义检索配置
ssr1 = SingleSearchRequest(
    anns_field="questions_dense",
    limit=30
)
ssr2 = SingleSearchRequest(
    anns_field="chunk_sparse",
    limit=20
)

fuse = FusionSpec(method="rrf", k=50)

search_config = SearchRequest(
    requests=[ssr1, ssr2],
    fuse=fuse,
    top_k=50,
    limit=5
)

# 应用新配置
rag.update_search_config(search_config)
```
