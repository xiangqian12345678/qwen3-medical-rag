# 多轮对话Agent

## 概述

多轮对话Agent是一个基于LangGraph的医疗问答智能体，支持多轮对话、主动追问、查询拆分和并行检索等高级功能。

## 功能特性

1. **主动追问**: 根据用户输入判断是否需要追问关键信息
2. **背景信息抽取**: 从多轮对话中提取和整合背景信息
3. **查询拆分**: 自动判断是否需要将复杂问题拆分为多个子查询
4. **并行检索**: 使用线程池并行执行多个子查询的RAG检索
5. **质量判断**: 支持RAG结果质量评估和重试机制

## 工程设计架构

### 整体架构

```
8.多轮对话Agent/
├── config/              # 配置模块
│   ├── __init__.py
│   ├── loader.py       # 配置加载器
│   └── models.py       # 配置数据模型
├── prompts/            # Prompt模板模块
│   ├── __init__.py
│   └── templates.py    # Prompt模板定义
├── agent/              # Agent模块
│   ├── __init__.py
│   ├── multi_dialogue_agent.py   # 多轮对话Agent主类
│   ├── search_graph.py           # 搜索图（单查询RAG）
│   ├── tools.py                 # Agent工具集
│   ├── utils.py                 # 工具函数（LLM/Embedding客户端创建）
│   ├── milvus/                  # Milvus检索模块
│   │   └── db_factory.py        # 知识库工厂
│   ├── search/                  # 网络检索模块
│   └── kgraph/                  # 知识图谱检索模块
├── multi_dialogue_agent.yaml    # 配置文件
├── run_example.py              # 示例代码
└── README.md                   # 本文档
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `config/` | 配置管理，支持YAML加载和动态修改 |
| `core/` | 核心工具，LLM/Embedding客户端创建、知识库管理 |
| `prompts/` | Prompt模板管理，统一管理所有LLM提示词 |
| `agent/` | Agent核心逻辑，包括多轮对话、搜索图、工具等 |

## 代码架构

### 多轮对话Agent状态流转

```
用户输入
    ↓
[ask] 判断是否需要追问 ──────────────┐
    ↓                             │
需要追问? → 是 → [END]            │
    ↓ 否                            │
[extract] 抽取背景信息               │
    ↓                             │
[split] 判断是否拆分子查询            │
    ↓                             │
[run] 并行执行多个SearchGraph       │
    ↓                             │
[answer] 汇总结果，写入对话历史         │
    ↓                             │
返回结果 ──────────────────────────┘
```

### SearchGraph状态流转（单查询RAG）

```
查询输入
    ↓
[db_search] 向量数据库检索
    ↓
[web_search] 网络检索 (可选)
    ↓
[rag] RAG生成回答
    ↓
[judge] 质量判断 (analysis模式) ──────┐
    ↓                                │
通过? → 否 → 重试(retry)              │
    ↓ 是                              │
[finish_success] 成功结束              │
                                    │
失败 → [finish_fail] 失败结束 ────────┘
```

### 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                     MultiDialogueAgent                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           MedicalAgentState (顶层状态)               │  │
│  │  - dialogue_messages: 主对话历史                     │  │
│  │  - asking_messages: 追问对话历史                   │  │
│  │  - background_info: 背景信息                        │  │
│  │  - sub_query: 子查询规划                            │  │
│  │  - sub_query_results: 子查询结果列表                │  │
│  └──────────────────────────────────────────────────────┘  │
│                        │                                   │
│                        ↓ 并行执行                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         SearchGraph 1        SearchGraph N          │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │       SearchMessagesState (子状态)             │  │  │
│  │  │  - query: 子查询                               │  │  │
│  │  │  - docs: 检索到的文档                          │  │  │
│  │  │  - answer: RAG回答                             │  │  │
│  │  │  - final: 最终结果                             │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                        │                                   │
│                        ↓ 汇总                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              final_answer                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 使用说明

### 基本使用

```python
from config.loader import ConfigLoader
from agent import MultiDialogueAgent, create_llm_client

# 加载配置
config_manager = ConfigLoader()
config = config_manager.appConfig

# 创建LLM客户端
power_model = create_llm_client(config.llm_config)

# 初始化Agent
agent = MultiDialogueAgent(config, power_model=power_model)

# 第一轮对话
state = agent.generate_answer("我这两天肚子痛")

# 如果需要追问
if state["ask_obj"].need_ask:
    print(f"需要追问: {state['ask_obj'].questions}")
    # 获取用户回复后继续
    state = agent.generate_answer("肚子疼得厉害，从昨天开始")

# 获取最终答案
print(state["final_answer"])
```

### 配置说明

配置文件 `rag_config.yaml` 主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `agent.mode` | 运行模式: analysis/fast/normal | analysis |
| `agent.max_attempts` | RAG最大重试次数 | 2 |
| `agent.max_ask_num` | 最大追问轮次 | 5 |
| `agent.network_search_enabled` | 是否启用网络搜索 | false |
| `dialogue.console_debug` | 控制台调试开关 | true |

### 运行模式说明

- **analysis**: 深度分析模式，包含质量判断和重试机制
- **fast**: 快速响应模式，RAG生成后直接返回
- **normal**: 均衡模式

## Milvus索引字段信息

与6.多轮对话RAG保持一致的索引字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `chunk` | VARCHAR | 文本块内容 |
| `parent_chunk` | VARCHAR | 父文本块内容 |
| `summary` | VARCHAR | 摘要内容 |
| `questions` | VARCHAR | 问题内容 |
| `source` | VARCHAR | 数据来源 |
| `source_name` | VARCHAR | 数据来源名称 |
| `lt_doc_id` | VARCHAR | 长文档ID |
| `chunk_id` | INT | 文本块ID |
| `hash_id` | VARCHAR | 哈希ID |
| `chunk_dense` | FLOAT_VECTOR(1024) | 文本块稠密向量 |
| `parent_chunk_dense` | FLOAT_VECTOR(1024) | 父文本块稠密向量 |
| `questions_dense` | FLOAT_VECTOR(1024) | 问题稠密向量 |
| `chunk_sparse` | SPARSE_FLOAT_VECTOR | 文本块稀疏向量(BM25) |

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

使用 DashScope API 需要设置 `DASHSCOPE_API_KEY` 环境变量。

**方式一: 使用 .env 文件 (推荐)**

```bash
# 复制示例文件
cp .env.example .env

# 编辑 .env 文件,填入你的 API Key
DASHSCOPE_API_KEY=your_api_key_here
```

获取 API Key: https://dashscope.console.aliyun.com/apiKey

**方式二: 直接设置环境变量**

```bash
# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="your_api_key_here"

# Windows (CMD)
set DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 启动 Milvus 服务

确保 Milvus 服务在 `http://localhost:19530` 运行。

## 运行示例

```bash
python run_example.py
```

## 依赖说明

- LangChain: LLM框架
- LangGraph: 状态图框架
- PyMilvus: 向量数据库客户端
- Pydantic: 数据校验
- Ollama/OpenAI: LLM服务提供商
