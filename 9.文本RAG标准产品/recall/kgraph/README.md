# 知识图谱检索模块 (KGraph)

## 文件功能说明

### 核心模块
| 文件 | 功能 |
|------|------|
| `graph_searcher.py` | 图谱检索核心类，支持向量相似度检索、关键词检索、关系检索 |
| `kgraph_search.py` | 知识图谱检索工具封装，提供LangChain工具接口和Agent集成 |
| `neo4j_connection.py` | Neo4j数据库连接管理类，负责建立、维护和关闭数据库连接 |

### 配置与工具
| 文件 | 功能 |
|------|------|
| `kg_config.yaml` | 配置文件（Neo4j连接、LLM、Embedding等） |
| `kg_config.py` | 配置加载器 |
| `kg_schema.json` | 知识图谱Schema定义（实体类型、关系类型） |
| `kgraph_schema.py` | Schema管理类，提供实体/关系查询接口 |
| `kg_utils.py` | 工具函数（LLM客户端创建、JSON转换等） |
| `kg_templates.py` | Prompt模板管理 |

## 核心代码逻辑

### 1. GraphSearcher 类流程

```python
# 主要检索方法链路
search_graph_by_query(query_text)
  ├─ _extract_entities_from_query()  # 从查询中提取实体
  ├─ _search_similar_entities()       # 向量检索相似实体
  ├─ query_by_entities()              # 根据实体ID查询子图
  └─ _format_kg_results()             # 格式化结果并去重
```

**检索方式：**
- **向量检索**：基于embedding相似度计算，支持 OpenAI/Ollama/DashScope
- **关键词检索**：Neo4j Cypher 查询，匹配节点名称
- **关系检索**：查询实体的所有关联关系

### 2. Agent 集成流程

```
用户查询 → LLM判断是否调用工具 → 执行图谱检索 → 返回Document列表 → 后续RAG使用
```

**关键组件：**
- `llm_kgraph_search()`：检索节点，控制工具调用
- `create_kgraph_search_tool()`：创建LangChain工具和ToolNode
- `json_to_list_document()`：将ToolMessage转换为Document

### 3. 向量索引加载

初始化时从Neo4j加载实体嵌入向量，构建内存索引用于快速相似度计算。

## 测试执行

```bash
cd 9.文本RAG标准产品/kgraph
python kgraph_search.py
```

**测试示例：**
1. 关键词检索：`search_graph_by_query("房颤的治疗目的是什么？")`
2. 关系检索：`search_by_relation("阿司匹林")`
3. 综合检索：`search_by_keyword("糖尿病")`

## 其他说明

### 配置文件
```yaml
neo4j:
  uri: bolt://localhost:7687
  database: neo4j

embedding:
  provider: dashscope  # 支持 openai/ollama/dashscope
  model: text-embedding-v2

agent:
  kgraph_search_enabled: true
  kgraph_search_cnt: 10
```

### 依赖
- Neo4j 数据库（需提前启动）
- LangChain
- Embedding服务（Ollama/DashScope/OpenAI）
- NumPy、scikit-learn（向量计算）

### 特性
- 支持多Provider嵌入模型切换
- 结果自动去重
- 单例模式管理检索器实例
- 支持向量检索回退到关键词检索
