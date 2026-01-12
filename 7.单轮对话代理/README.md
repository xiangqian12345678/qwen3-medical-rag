# 单轮对话代理

单轮对话代理服务，基于RAG和LangGraph实现的医疗知识问答Agent。

## 功能特性

1. **混合检索**：支持稠密向量（Dense）和稀疏向量（Sparse/ BM25）的混合检索
2. **向量融合**：支持RRF（倒数排名融合）和Weighted（加权融合）两种融合策略
3. **Agent模式**：
   - `analysis`：分析模式，带质量判断和重试机制
   - `fast`：快速模式，直接返回结果
   - `normal`：普通模式
4. **工具调用**：支持向量数据库检索工具

## 目录结构

```
7.单轮对话代理/
├── config/
│   ├── __init__.py
│   ├── loader.py          # 配置加载器
│   └── models.py          # 配置数据模型
├── core/
│   ├── __init__.py
│   ├── utils.py           # 工具函数
│   └── db_factory.py      # 知识库工厂
├── prompts/
│   ├── __init__.py
│   └── templates.py       # 提示词模板
├── agent/
│   ├── __init__.py
│   ├── tools.py           # Agent工具类
│   └── search_agent.py    # 单轮对话Agent
├── single_dialogue_agent.yaml  # 配置文件
├── __init__.py
└── run_example.py         # 示例代码
```

## 配置说明

配置文件 `single_dialogue_agent.yaml` 包含以下配置项：

### Milvus配置
```yaml
milvus:
  uri: http://localhost:19530
  collection_name: medical_knowledge
```

### 嵌入配置
```yaml
embedding:
  summary_dense:    # 问题向量
    provider: ollama
    model: bge-m3:latest
  text_dense:       # 答案向量
    provider: ollama
    model: bge-m3:latest
  text_sparse:      # BM25稀疏向量
    provider: self
    vocab_path_or_name: ../output/vocab/vocab.pkl.gz
```

### Agent配置
```yaml
agent:
  mode: analysis           # 运行模式：analysis/fast/normal
  max_attempts: 2          # 最大重试次数
  network_search_enabled: false  # 是否启用联网搜索
  console_debug: false     # 是否输出调试日志
```

## 使用示例

### 基本使用

```python
from 单轮对话代理 import ConfigLoader, SingleDialogueAgent, create_llm_client

# 加载配置
config_loader = ConfigLoader()
config = config_loader.config

# 创建LLM客户端
power_llm = create_llm_client(config.llm)

# 创建Agent
agent = SingleDialogueAgent(config, power_llm)

# 提问
answer = agent.answer("高血压的症状有哪些？")
print(answer)
```

### 自定义配置路径

```python
config_loader = ConfigLoader("/path/to/config.yaml")
```

### 动态修改配置

```python
config_loader.change({
    "agent.mode": "fast",
    "agent.console_debug": True
})
```

## 运行示例

```bash
cd "7.单轮对话代理"
python run_example.py
```

## 字段配置说明

与 `5.单轮对话RAG` 保持一致的字段映射：

| 配置字段 | 数据库字段 | 说明 |
|---------|----------|------|
| summary_field | question | 问题字段 |
| document_field | answer | 答案字段 |
| source_field | source | 数据来源 |
| source_name_field | source_name | 数据源名称 |

## 依赖要求

- langchain
- langchain-core
- langchain-openai
- langchain-ollama
- pymilvus
- pydantic
- pyyaml
