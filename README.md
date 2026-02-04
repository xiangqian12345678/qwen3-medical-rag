# Qwen3 医疗大模型 RAG 系统

基于 LangChain、LangGraph、Milvus 和 Neo4j 的医疗领域检索增强生成（RAG）系统，提供从数据处理到智能问答的完整解决方案。

---

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [代码结构图](#代码结构图)
- [开发进度](#开发进度)
- [常见问题](#常见问题)

---

## 项目概述

本项目是一个专业的医疗知识问答系统，通过结合大语言模型（LLM）、向量数据库、知识图谱和网络搜索，为用户提供准确、专业的医疗知识问答服务。项目采用模块化设计，从基础数据处理到完整的RAG应用，提供全方位的技术支持和解决方案。

### 项目亮点

- **全流程覆盖**：从数据采集、清洗、标注到索引构建、知识图谱、RAG问答，完整覆盖数据处理全链路
- **模块化设计**：9个独立模块，可按需组合使用，灵活适配不同业务场景
- **配置驱动**：核心功能均通过配置文件管理，无需修改代码即可调整策略
- **生产级架构**：支持多种LLM提供商、多种检索方式、多种融合策略，满足生产环境需求

---

## 核心特性

### 🔍 多源融合检索

- **向量检索**：支持稠密向量（Dense）和稀疏向量（BM25）混合检索
- **知识图谱**：基于 Neo4j 的实体关系检索和向量相似度检索
- **网络搜索**：实时获取最新医疗资讯（DuckDuckGo）
- **多路融合**：RRF 和加权融合算法优化检索结果

### 🧠 智能查询增强

- **主动追问**：多轮对话中智能补充关键信息
- **Query 改写**：将口语化问题转化为专业检索词
- **多查询生成**：从不同角度生成多个检索查询
- **子查询拆分**：将复杂问题拆解为多个独立子问题
- **上位词生成**：生成更通用的上位问题（Step-back Prompting）
- **假设性答案**：基于问题生成假设答案用于检索（HyDE）

### 📄 文档过滤与排序

- **相关性过滤**：基于 Embedding 或 LLM 过滤低相关文档
- **内容压缩**：提取与问题最相关的内容片段
- **冗余去除**：基于向量相似度去除重复文档
- **重排序**：Cross-Encoder 和长上下文重排序

### 💬 多轮对话管理

- **对话历史**：完整的多轮对话上下文管理
- **背景提取**：从对话中自动抽取关键背景信息
- **追问机制**：智能判断是否需要追问
- **摘要缓存**：高效的对话摘要管理，支持历史压缩

---

## 技术栈

| 分类        | 技术                       | 说明         |
|-----------|--------------------------|------------|
| **框架**    | LangChain、LangGraph      | LLM 应用开发框架 |
| **LLM**   | 通义千问、GPT、Ollama          | 大语言模型      |
| **向量数据库** | Milvus                   | 高性能向量检索    |
| **图数据库**  | Neo4j                    | 知识图谱存储与查询  |
| **嵌入模型**  | BGE-M3、text-embedding-v2 | 文本向量化      |
| **重排序**   | text-reranker-v2         | 检索结果重排序    |
| **中文分词**  | pkuseg                   | 专业中文分词     |
| **搜索引擎**  | DuckDuckGo (ddgs)        | 网络搜索       |

## 大模型微调

[基于qwen3构建领域大模型](https://github.com/xiangqian12345678/qwen3-medical)


---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Qwen3 医疗 RAG 系统                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ 0.基础知识   │    │ 1.数据处理   │    │ 2.词库生成   │                  │
│  │ RAG原理学习  │───→│ 数据融合/清理 │───→│ BM25词表构建 │                  │
│  │ 索引优化技巧  │    │ 数据过滤/标注 │    │ 中文分词     │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                  │                  │                              │
│         ▼                  ▼                  ▼                              │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │                    数据层                              │                  │
│  │  data/ - 原始数据 | output/ - 处理结果 | model/ - 模型 │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                              │                                                 │
│                              ▼                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ 3.索引构建   │    │ 4.知识图谱   │    │ 5.单轮RAG   │                  │
│  │ Milvus向量库 │    │ Neo4j图谱库  │    │ 基础问答系统 │                  │
│  │ 混合检索     │    │ 实体关系抽取 │    │ 混合检索     │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                  │                  │                              │
│         └──────────────────┴──────────────────┘                              │
│                              │                                                 │
│                              ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │                    Agent 层                           │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                              │                                                 │
│         ┌────────────────────┼────────────────────┐                        │
│         ▼                    ▼                    ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │ 6.多轮RAG   │    │ 7.单轮Agent  │    │ 8.多轮Agent  │                │
│  │ 对话历史管理 │    │ LangGraph    │    │ 主动追问     │                │
│  │ 查询改写     │    │ 质量判断     │    │ 子查询拆分   │                │
│  └──────────────┘    └──────────────┘    └──────────────┘                │
│                              │                                                 │
│                              ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │              9.文本RAG标准产品                         │                  │
│  │  配置化 RAG 框架 | 多源检索融合 | 全链路优化          │                  │
│  └──────────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RAG 标准产品流程图

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  主动追问 (ask_user)                                    │
│  - 判断是否需要追问关键信息                              │
│  - 追问对话历史管理                                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼ (无需追问)
┌─────────────────────────────────────────────────────────┐
│  查询增强 (query_enhance)                               │
│  - 从对话中提取背景信息                                  │
│  - 查询改写（口语化 → 专业化）                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  召回增强 (recall_enhance)                             │
│  - 多查询生成（不同角度）                                │
│  - 子查询拆分（复杂问题）                                │
│  - 上位词生成（Step-back）                               │
│  - 假设性答案生成（HyDE）                                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  文档召回 (recall)                                      │
│  ┌─────────────┬─────────────┬─────────────┐          │
│  │ Milvus向量  │ Neo4j图谱   │ DuckDuckGo  │          │
│  │ 稠密+稀疏   │ 实体+关系   │ 网络搜索    │          │
│  └─────────────┴─────────────┴─────────────┘          │
│              并行召回 → 结果融合                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  过滤增强 (filter_enhance)                              │
│  - 基于LLM内容相关性过滤                                 │
│  - 基于Embedding相关性过滤                              │
│  - 冗余文档去除                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  排序增强 (sort_enhance)                                │
│  - Cross-Encoder 重排序                                 │
│  - 长上下文重排序                                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  答案生成 (answer)                                      │
│  - 基于检索文档生成答案                                 │
│  - 明确说明无法回答的问题                               │
│  - 提供诊疗建议提醒                                     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
  返回答案
```

---

## 快速开始

### 环境要求

- Python 3.8+
- Docker Desktop
- 8GB+ 内存
- 20GB+ 磁盘空间

### 安装依赖

```bash
# 1. 创建虚拟环境
conda create -n medicalrag python=3.11
conda activate medicalrag

# 2. 安装 Python 依赖
pip install -r requirements.txt
```

### 基础服务部署

#### 1.安装docker desktop

    下载地址：  https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
    安装测试：  docker --version (powershell)

#### 2. Milvus 向量数据库

```bash
# 下载配置文件
curl -O https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml

# 启动 Milvus
docker compose up -d

# 验证服务
docker ps
```

#### 3. Neo4j 图数据库

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/12345678 \
  neo4j:latest
```

#### 4. Ollama 本地模型

```bash
# 下载并安装 Ollama
# https://ollama.ai/download

# 拉取所需模型
ollama pull bge-m3:latest        # 嵌入模型
ollama pull qwen3:4b             # 对话模型
ollama pull dengcao/bge-reranker-v2-m3  # 重排序模型
```

### 数据准备与索引构建

```bash
# 1. 数据处理（格式统一、清理、过滤、标注）
cd 1.数据处理
python data-clean.py

# 2. 生成 BM25 词表
cd ../2.词库生成
python build_vocab.py

# 3. 构建 Milvus 索引
cd ../3.索引构建
python run_build_index.py

# 4. 构建知识图谱（可选）
cd ../4.知识图谱
python run_create_graph.py
```

### 运行 RAG 系统

```bash
# 单轮对话 RAG
cd 5.单轮对话RAG
python run_example.py

# 多轮对话 RAG
cd ../6.多轮对话RAG
python run_example.py

# 单轮对话 Agent
cd ../7.单轮对话Agent
python run_example.py

# 多轮对话 Agent
cd ../8.多轮对话Agent
python run_example.py

# 完整 RAG 标准产品
cd ../9.文本RAG标准产品
python run_dialogue.py
```

---

## 模块说明

### 0. 基础知识

RAG 技术原理学习和基础示例代码，包含：

- 索引优化：摘要索引、父子索引、假设性问题索引、元数据索引
- 召回增强：Query 优化、Multi-query、问题分解、Step-back、HyDE
- 排序过滤：RRF、Cross-Encoder、长上下文重排序、LLM 过滤、冗余去除
- 多模态 RAG：基于多模态向量模型和大模型的实现

**适用场景**：学习 RAG 技术原理，了解各项优化技术

---

### 1. 数据处理

数据处理全流程，包括数据融合、清理、过滤和标注。

**核心功能**：

- 数据融合：统一不同来源的数据格式（PDF/TXT/JSONL）
- 数据清理：消除低质量内容（连贯性、实用性、安全性、清晰度）
- 数据过滤：过滤政治、色情、非目标语义内容
- 数据标注：主题、风格、实体、摘要、关键词标注

**输出**：标准化的 JSONL 格式数据，存储在 `output/annotation` 目录

---

### 2. 词库生成

从标注数据构建 BM25 检索所需的词表。

**核心功能**：

- 支持并行分词处理
- 中文专业分词（基于 pkuseg）
- BM25 参数计算（TF-IDF）
- gzip 压缩存储

**输出**：`output/vocab/vocab.pkl.gz`

**使用方法**：

```bash
cd 2.词库生成
python build_vocab.py
```

---

### 3. 索引构建

医疗知识库索引构建和检索模块，基于 Milvus 实现混合向量检索。

**核心特性**：

- 多向量字段支持：支持多个稠密向量字段和稀疏向量字段
- List 类型字段展开：对于 `questions` 等列表类型字段，每个元素会展开为单独的向量索引行
- 向量融合检索：支持 RRF 和加权融合策略
- 配置驱动：所有配置通过 `index.yaml` 管理

**多向量字段展开示例**：

```
输入文档：{"chunk": "...", "questions": ["问题1", "问题2"]}

展开后：
- 行1: questions_dense="问题1", origin_pk=abc, vector_id=0
- 行2: questions_dense="问题2", origin_pk=abc, vector_id=1
- 行3: chunk_dense="...", origin_pk=abc, vector_id=-1
```

**索引字段**：
| 字段 | 类型 | 说明 |
|------|------|------|
| chunk | VARCHAR | 文本块内容 |
| parent_chunk | VARCHAR | 父文本块内容 |
| summary | VARCHAR | 摘要内容 |
| questions | VARCHAR | 问题内容 |
| chunk_dense | FLOAT_VECTOR(1024) | 文本块稠密向量 |
| parent_chunk_dense | FLOAT_VECTOR(1024) | 父文本块稠密向量 |
| questions_dense | FLOAT_VECTOR(1024) | 问题稠密向量 |
| chunk_sparse | SPARSE_FLOAT_VECTOR | 文本块稀疏向量(BM25) |

**使用方法**：

```bash
# 构建索引
python run_build_index.py

# 清理索引
python run_clean_index.py

# 测试检索
python test_dashscope.py
```

---

### 4. 知识图谱

基于 Neo4j 的知识图谱 RAG 系统，支持单轮会话，整合向量检索和知识图谱检索。

**核心功能**：

- 完整的 Neo4j 操作：实体和关系的 CRUD 操作
- 大模型集成：使用通义千问进行实体关系提取和答案生成
- 向量检索：基于嵌入向量的相似度检索
- 图查询：关键字搜索、最短路径、子图查询等高级功能
- RAG 整合：结合知识图谱和向量检索生成准确答案

**知识图谱 Schema**：

```json
{
  "entity_types": [
    "药物",
    "症状",
    "疾病"
  ],
  "relationship_types": [
    {
      "name": "治疗",
      "source": [
        "药物"
      ],
      "target": [
        "症状",
        "疾病"
      ]
    },
    {
      "name": "导致",
      "source": [
        "疾病"
      ],
      "target": [
        "症状"
      ]
    },
    {
      "name": "属于",
      "source": [
        "症状"
      ],
      "target": [
        "疾病"
      ]
    }
  ]
}
```

**使用方法**：

```bash
# 创建知识图谱
python run_create_graph.py

# 运行 RAG 查询
python run_rag.py

# 清理数据库
python run_clean_neo4j.py

# 测试基础功能
python test_basic_function.py
```

---

### 5. 单轮对话 RAG

医疗知识库检索增强生成(RAG)系统，支持混合检索和向量融合。

**核心功能**：

- 混合检索：稠密向量（chunk/parent_chunk/questions）+ 稀疏向量（BM25）
- 向量融合：RRF 或 Weighted 融合策略
- LangChain 集成：标准 LangChain 检索器接口

**检索配置**（`simple_rag.yaml`）：

```yaml
dense_fields:
  chunk:
    provider: ollama
    model: bge-m3:latest
    dimension: 1024
  parent_chunk:
    provider: ollama
    model: bge-m3:latest
  questions:
    provider: ollama
    model: bge-m3:latest

sparse_fields:
  chunk:
    provider: self
    vocab_path_or_name: ../output/vocab/vocab.pkl.gz
    k1: 1.5
    b: 0.75

rag:
  default_fields:
    - anns_field: chunk_dense
    - anns_field: parent_chunk_dense
    - anns_field: questions_dense
    - anns_field: chunk_sparse
  fusion:
    method: rrf
    k: 60
  limit: 5
```

**使用方法**：

```bash
python run_example.py
```

---

### 6. 多轮对话 RAG

多轮对话医疗 RAG 系统，支持会话管理、历史压缩、查询改写、混合检索等功能。

**核心功能**：

- 会话隔离：通过 `session_id` 区分不同用户会话
- 历史压缩：自动检测对话长度，生成摘要释放 token 空间
- 查询改写：基于对话历史将当前问题改写为独立查询
- 动态上下文：根据 token 预算动态调整文档上下文内容
- Token 估算：支持多种 token 估算方法（avg/tiktoken）

**处理流程**：

```
用户问题
  ↓
检查对话历史是否过长
  ↓ (超限)
压缩旧对话为摘要 → 保留最新对话
  ↓
查询改写（用户问题 + 对话历史 → 独立查询）
  ↓
混合检索（稠密 + 稀疏 → RRF/Weighted 融合）
  ↓
动态文档上下文构建（根据剩余 token 截取文档）
  ↓
答案生成（系统prompt + 运行摘要 + 对话历史 + 文档上下文 + 用户问题）
```

**使用方法**：

```bash
python run_example.py
```

**代码集成示例**：

```python
from config import ConfigLoader
from multi_dialogue_rag import MultiDialogueRag

config_loader = ConfigLoader()
rag = MultiDialogueRag(config_loader.config)

# 提问（指定 session_id 维护会话上下文）
result = rag.generate_answer(
    query="什么是高血压？",
    session_id="user_001",
    return_document=True
)
```

---

### 7. 单轮对话 Agent

单轮对话代理服务，基于 RAG 和 LangGraph 实现的医疗知识问答 Agent。

**核心功能**：

- 混合检索：稠密向量 + 稀疏向量（BM25）
- 向量融合：RRF 和 Weighted 融合策略
- Agent 模式：
    - `analysis`：分析模式，带质量判断和重试机制
    - `fast`：快速模式，直接返回结果
    - `normal`：普通模式
- 工具调用：支持向量数据库检索工具

**Agent 状态流转**：

```
查询输入
  ↓
[db_search] 向量数据库检索
  ↓
[web_search] 网络检索 (可选)
  ↓
[rag] RAG 生成回答
  ↓
[judge] 质量判断 (analysis 模式)
  ↓ (不通过)
重试 (retry)
  ↓ (通过)
[finish_success] 成功结束
```

**使用方法**：

```bash
python run_example.py
```

---

### 8. 多轮对话 Agent

多轮对话 Agent 是一个基于 LangGraph 的医疗问答智能体，支持多轮对话、主动追问、查询拆分和并行检索等高级功能。

**核心功能**：

- 主动追问：根据用户输入判断是否需要追问关键信息
- 背景信息抽取：从多轮对话中提取和整合背景信息
- 查询拆分：自动判断是否需要将复杂问题拆分为多个子查询
- 并行检索：使用线程池并行执行多个子查询的 RAG 检索
- 质量判断：支持 RAG 结果质量评估和重试机制

**Agent 状态流转**：

```
用户输入
  ↓
[ask] 判断是否需要追问 ─────────┐
  ↓                              │
需要追问? → 是 → [END]           │
  ↓ 否                           │
[extract] 抽取背景信息            │
  ↓                              │
[split] 判断是否拆分子查询         │
  ↓                              │
[run] 并行执行多个 SearchGraph    │
  ↓                              │
[answer] 汇总结果，写入对话历史     │
  ↓                              │
返回结果 ─────────────────────────┘
```

**SearchGraph 状态流转（单查询 RAG）**：

```
查询输入
  ↓
[db_search] 向量数据库检索
  ↓
[web_search] 网络检索 (可选)
  ↓
[rag] RAG 生成回答
  ↓
[judge] 质量判断 (analysis 模式) ─┐
  ↓                               │
通过? → 否 → 重试 (retry)          │
  ↓ 是                            │
[finish_success] 成功结束          │

失败 → [finish_fail] 失败结束 ────┘
```

**使用方法**：

```bash
python run_example.py
```

---

### 9. 文本 RAG 标准产品

基于 LangGraph 和 LangChain 的医学领域检索增强生成（RAG）对话系统，支持多轮对话、复杂查询增强和多种检索方式融合。

**核心功能**：

- 多源融合检索：向量检索、知识图谱、网络搜索
- 智能查询增强：主动追问、Query 改写、多查询生成、子查询拆分、上位词生成、假设性答案
- 文档过滤与排序：相关性过滤、内容压缩、冗余去除、重排序
- 多轮对话管理：对话历史、背景提取、追问机制、摘要缓存

**完整流程**：

```
用户输入
  ↓
主动追问 (ask_user)
  ↓
查询增强 (query_enhance) - 背景/改写
  ↓
召回增强 (recall_enhance) - 多查询/子查询/上位词/HyDE
  ↓
文档召回 (recall) - Milvus/Neo4j/Web 并行召回
  ↓
过滤增强 (filter_enhance) - 相关性过滤/冗余去除
  ↓
排序增强 (sort_enhance) - Cross-Encoder/长上下文
  ↓
答案生成 (answer)
  ↓
返回答案
```

**配置文件**：

- `rag/rag_config.yaml`：LLM、Embedding、Reranker、Agent 配置
- `recall/milvus/embed_config.yaml`：Milvus 向量检索配置
- `recall/kgraph/kg_config.yaml`：Neo4j 知识图谱配置

**使用方法**：

```bash
# 启动交互式对话
python run_dialogue.py

# 测试各模块
python test_milvus.py    # 测试向量检索
python test_kgraph.py    # 测试知识图谱
python test_search.py    # 测试网络搜索
```

---

## 代码结构图

### 完整项目结构

```
qwen3-medical-rag/
├── 0.基础知识/                              # RAG 原理学习和示例
│   ├── 1.索引生成/                          # 索引优化技术
│   │   ├── 1.索引优化-摘要索引.py
│   │   ├── 2.索引优化-父子索引.py
│   │   ├── 3.索引优化-假设性问题索引.py
│   │   └── 4.索引优化-元数据.py
│   ├── 6.召回增强/                          # 召回增强技术
│   │   ├── 6.查询优化-Enrich完善用户问题.py
│   │   ├── 7.查询优化-Mulit-query多路召回.py
│   │   ├── 8.查询优化-问题分解.py
│   │   ├── 9.查询优化-上位优化-stepback.py
│   │   ├── 10.查询优化-假设性文档嵌入-HyDE.py
│   │   └── 11.查询优化-混合检索.py
│   ├── 7.排序过滤/                          # 排序过滤技术
│   │   ├── 1.重排序-RRF算法.py
│   │   ├── 2.重排序-RRF算法.py
│   │   ├── 3.重排序-模型排序-CrossEncoder.py
│   │   ├── 4.重排序-模型排序-CrossEncoderReranker.py
│   │   ├── 5.重排序-LongContextReorder.py
│   │   ├── 6.压缩过滤-LLMChainExtractor.py
│   │   ├── 7.压缩过滤-LLMChainFilter.py
│   │   ├── 8.压缩过滤-EmbeddingsFilter.py
│   │   └── 9.冗余过滤-EmbeddingsRedundantFilter.py
│   └── 8.多模态RAG/                         # 多模态 RAG 示例
│       ├── 1.基于多模态向量模型.py
│       ├── 2.基于多模态大模型.py
│       └── 3.基于多模态大模型（平衡成本与效果）.py
│
├── 1.数据处理/                              # 数据处理全流程
│   ├── 1.1 数据融合/
│   │   ├── dataFusion.py                   # 数据融合主程序
│   │   ├── jsonl2jsonl.py
│   │   ├── pdf2jsonl.py
│   │   └── txt2jsonl.py
│   ├── 1.2 数据清理/
│   │   └── data-clean.py                   # 数据清理
│   ├── 1.3 数据过滤/
│   │   └── data_filter.py                  # 数据过滤
│   ├── 1.4 数据标注/
│   │   └── dataAnnotation.py                # 数据标注
│   └── README.md
│
├── 2.词库生成/                              # BM25 词表生成
│   ├── vocabulary.py                        # 词表管理类
│   ├── tokenizer.py                         # 分词器（支持并行）
│   ├── build_vocab.py                       # 词库构建主程序
│   └── README.md
│
├── 3.索引构建/                              # Milvus 索引构建
│   ├── config.py                            # 配置加载和模型定义
│   ├── embedding_client.py                  # 嵌入模型客户端工厂
│   ├── vocab.py                             # 词表管理（稀疏向量）
│   ├── sparse_vectorizer.py                # 稀疏向量处理器（BM25）
│   ├── vectorizer.py                        # 文档向量化器（多向量字段展开）
│   ├── collection.py                        # Milvus Collection 管理器
│   ├── searcher.py                          # 检索器（混合检索和向量融合）
│   ├── insert.py                            # 数据插入工具
│   ├── knowledge_base.py                    # 知识库主类
│   ├── ingest.py                            # 数据导入工具
│   ├── index.yaml                           # 索引配置文件
│   ├── run_build_index.py                   # 索引构建脚本
│   ├── run_clean_index.py                  # 索引清理脚本
│   └── README.md
│
├── 4.知识图谱/                              # Neo4j 知识图谱
│   ├── config/
│   │   ├── kg_config.yaml                   # 知识图谱配置
│   │   └── kg_schema.json                  # 知识图谱 Schema
│   ├── config.py                            # 配置管理模块
│   ├── neo4j_connection.py                  # Neo4j 连接管理
│   ├── embedding_service.py                 # 嵌入向量生成服务
│   ├── llm_service.py                       # 大模型服务
│   ├── neo4j_operations.py                 # 基础数据库 CRUD 操作
│   ├── neo4j_query.py                       # 图查询功能
│   ├── neo4j_save.py                        # 数据保存功能
│   ├── vector_search.py                     # 向量相似度检索
│   ├── rag_system.py                        # RAG 系统整合
│   ├── text_processor.py                    # 文本处理功能
│   ├── run_create_graph.py                 # 创建知识图谱主程序
│   ├── run_rag.py                           # 运行 RAG 查询
│   ├── run_clean_neo4j.py                  # Neo4j 数据库清理工具
│   ├── test_basic_function.py               # 基础功能测试
│   └── README.md
│
├── 5.单轮对话RAG/                           # 单轮 RAG 系统
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py                        # 配置加载器
│   │   └── models.py                        # 配置数据模型
│   ├── embed/
│   │   ├── __init__.py
│   │   ├── vocab.py                         # 词表管理
│   │   └── bm25.py                          # BM25 稀疏向量
│   ├── knowledge_base.py                    # 知识库核心类
│   ├── retriever.py                         # LangChain 检索器
│   ├── single_dialogue_rag.py               # RAG 系统
│   ├── utils.py                             # 工具函数
│   ├── prompts.py                           # Prompt 模板
│   ├── simple_rag.yaml                      # 配置文件
│   ├── run_example.py                       # 运行示例
│   └── README.md
│
├── 6.多轮对话RAG/                           # 多轮 RAG 系统
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── models.py
│   ├── embed/
│   │   ├── __init__.py
│   │   ├── vocab.py
│   │   └── bm25.py
│   ├── knowledge_base.py
│   ├── retriever.py
│   ├── multi_dialogue_rag.py                # 多轮对话 RAG 核心
│   ├── rag_base.py                          # RAG 抽象基类
│   ├── utils.py
│   ├── prompts.py
│   ├── multi_dialogue.yaml                  # 配置文件
│   ├── run_example.py
│   └── README.md
│
├── 7.单轮对话Agent/                         # 单轮 Agent
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── models.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   └── db_factory.py                   # 知识库工厂
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── templates.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py
│   │   └── search_agent.py                  # 单轮对话 Agent
│   ├── single_dialogue_agent.yaml          # 配置文件
│   ├── run_example.py
│   └── README.md
│
├── 8.多轮对话Agent/                         # 多轮 Agent
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── models.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   └── db_factory.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── templates.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── multi_dialogue_agent.py          # 多轮对话 Agent 主类
│   │   ├── search_graph.py                  # 搜索图（单查询 RAG）
│   │   ├── tools.py
│   │   └── utils.py
│   ├── multi_dialogue_agent.yaml            # 配置文件
│   ├── run_example.py
│   └── README.md
│
├── 9.文本RAG标准产品/                       # 完整 RAG 产品
│   ├── app_config.py                        # 应用配置管理
│   ├── dialogue_agent.py                    # 对话 Agent 核心
│   ├── integrated_recall.py                 # 集成检索模块
│   ├── templates.py                         # 全局 Prompt 模板
│   ├── utils.py                             # 工具函数
│   ├── answer/                              # 答案生成模块
│   │   ├── __init__.py
│   │   ├── answer.py                        # 答案生成核心逻辑
│   │   ├── answer_templates.py              # RAG 提示模板
│   │   └── utils.py
│   ├── enhance/                             # 增强模块
│   │   ├── __init__.py
│   │   ├── agent_state.py                   # Agent 状态定义
│   │   ├── query_enhance.py                 # 查询增强
│   │   ├── recall_enhance.py                # 召回增强
│   │   ├── filter_enhance.py                # 过滤增强
│   │   ├── sort_enhance.py                  # 排序增强
│   │   ├── enhance_templates.py             # 增强提示模板
│   │   └── utils.py
│   ├── rag/                                 # RAG 配置模块
│   │   ├── __init__.py
│   │   ├── rag_config.py                    # 配置数据模型
│   │   ├── rag_config.yaml                  # 配置文件
│   │   └── rag_loader.py                    # 配置加载器
│   ├── recall/                              # 检索模块
│   │   ├── milvus/                          # 向量检索
│   │   │   ├── embed_config.py
│   │   │   ├── embed_config.yaml
│   │   │   ├── embed_loader.py
│   │   │   ├── embed_searcher.py            # 核心检索器
│   │   │   ├── embed_search.py              # LangChain 集成
│   │   │   ├── sparse_vectorizer.py         # BM25 实现
│   │   │   └── README.md
│   │   ├── kgraph/                          # 知识图谱检索
│   │   │   ├── kg_config.py
│   │   │   ├── kg_config.yaml
│   │   │   ├── kgraph_searcher.py           # 核心检索器
│   │   │   ├── kgraph_search.py             # LangChain 集成
│   │   │   ├── neo4j_connection.py          # Neo4j 连接管理
│   │   │   └── README.md
│   │   ├── search/                          # 网络搜索
│   │   │   ├── web_searcher.py              # 核心检索器
│   │   │   ├── web_search.py                # LangChain 集成
│   │   │   └── search_utils.py
│   │   └── __init__.py
│   ├── test_*.py                            # 测试文件
│   ├── run_dialogue.py                      # 主程序入口
│   └── README.md
│
├── configuration/                           # 全局配置
├── data/                                    # 原始数据
│   ├── annotation/                          # 标注数据
│   ├── graph/                               # 知识图谱数据
│   └── raw/                                 # 原始数据
├── model/                                   # 模型存储
├── output/                                  # 输出数据
│   ├── annotation/                          # 数据标注输出
│   ├── vocab/                               # 词表输出
│   └── graph/                               # 知识图谱输出
├── requirements.txt                         # Python 依赖
├── 工作规划.md                              # 项目规划文档
└── README.md                                # 本文档
```

---

## 开发进度

| 模块             | 进度   | 说明                |
|----------------|------|-------------------|
| 0. 基础知识        | 100% | RAG 原理学习完成        |
| 1. 数据处理        | 90%  | 核心算法完善            |
| 2. 词库生成        | 90%  | 停用词缺失             |
| 3. 索引构建        | 90%  | 配置无用字段清除          |
| 4. 知识图谱        | 100% | 功能完整              |
| 5. 单轮对话 RAG    | 100% | 检索字段配置、排序字段配置确认   |
| 6. 多轮对话 RAG    | 100% | 功能完整              |
| 7. 单轮对话 Agent  | 100% | 功能完整              |
| 8. 多轮对话 Agent  | 100% | 功能完整              |
| 9. 文本 RAG 标准产品 | 95%  | 全流程调试、代码简化、性能优化完成 |

### 待优化项

1. **多模态代理模式**：基于多轮对话 Agent 改造
2. **类命名规范化**：形成标准化的 RAG 架构命名

---

## 常见问题

### Q: 如何切换 LLM 提供商？

修改配置文件中的 `provider` 字段：

**DashScope（阿里云）**：

```yaml
llm:
  provider: dashscope
  api_key: your-api-key
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model: qwen-plus
```

**OpenAI**：

```yaml
llm:
  provider: openai
  api_key: your-api-key
  model: gpt-4
```

**Ollama（本地）**：

```yaml
llm:
  provider: ollama
  base_url: http://localhost:11434
  model: qwen3:4b
```

### Q: 如何调整召回数量？

修改配置文件中的 `limit` 和 `top_k` 参数：

```yaml
default_search:
  limit: 5     # 最终返回数量
  top_k: 50    # 每路检索返回数量（融合前）
```

### Q: 如何启用/禁用某种检索方式？

在 RAG 配置文件中修改对应开关：

```yaml
agent:
  network_search_enabled: false    # 禁用网络搜索
  kgraph_search_enabled: false     # 禁用知识图谱
```

### Q: 如何调整相关性阈值？

修改过滤配置中的阈值：

```yaml
agent:
  low_correction_threshold: 0.7    # 提高相关性要求
  redundant_threshold: 0.90       # 降低去重敏感度
```

### Q: 如何启用不同的融合策略？

修改向量融合配置：

**RRF 融合**：

```yaml
fusion:
  method: rrf
  k: 60
```

**加权融合**：

```yaml
fusion:
  method: weighted
  weights:
    chunk_dense: 0.35
    parent_chunk_dense: 0.35
    questions_dense: 0.20
    chunk_sparse: 0.10
```

### Q: 如何优化性能？

1. **并行检索**：启用多线程并行执行多个查询
2. **缓存**：启用对话摘要缓存
3. **索引优化**：选择合适的索引类型（HNSW/IVF_FLAT）和参数
4. **批量处理**：使用批量 API 减少网络请求
5. **召回数限制**：合理设置 `top_k` 避免过度召回

---

## 参考资源

- [Medical-Graph-RAG](https://github.com/ImprintLab/Medical-Graph-RAG)
- [OpenGraphRAG/MedicalGraphRAG](https://github.com/OpenGraphRAG/MedicalGraphRAG)
- [ai-medical-assistant](https://github.com/zhttyy520/ai-medical-assistant)

---

## 许可证

使用声明：随便学习下载和商业化

---

## 联系方式

微信： 13552482980
qq： 1012088761

## 参考代码

[medical-rag](https://github.com/ImprintLab/Medical-Graph-RAG)

[open-graph-rag](https://github.com/OpenGraphRAG/MedicalGraphRAG?tab=readme-ov-file)

[medical-graph-rag](https://github.com/OpenGraphRAG/MedicalGraphRAG)

[ai-medical-assistant](https://github.com/zhttyy520/ai-medical-assistant)