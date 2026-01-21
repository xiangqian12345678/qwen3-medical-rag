# 配置文件说明

## 配置文件拆分结构

项目配置已拆分为多个独立模块，每个模块包含配置文件和加载器：

### 1. 知识图谱模块 (`agent/kgraph/`)

- **配置文件**: `kg_config.yaml`
  - Neo4j数据库连接配置
  - LLM配置
  - 知识图谱搜索配置

- **配置类**: `kg_config.py`
  - `Neo4jConfig`: Neo4j连接配置
  - `LLMConfig`: LLM客户端配置
  - `KGraphAgentConfig`: 知识图谱Agent配置

- **加载器**: `kg_loader.py`
  - `KGraphConfigLoader`: 加载知识图谱配置

- **工具函数**: `kg_utils.py`
  - `create_llm_client()`: 创建LLM客户端

### 2. 向量检索模块 (`agent/milvus/`)

- **配置文件**: `embed_config.yaml`
  - Milvus数据库连接配置
  - Embedding配置（summary_dense, text_dense, text_sparse）

- **配置类**: `embed_config.py`
  - `MilvusConfig`: Milvus连接配置
  - `DenseConfig`: 稠密向量配置
  - `SparseConfig`: 稀疏向量配置（BM25）
  - `EmbeddingConfig`: 嵌入配置
  - `SearchRequest`: 检索请求配置

- **加载器**: `embed_loader.py`
  - `EmbedConfigLoader`: 加载向量检索配置

- **工具函数**: `embed_utils.py`
  - `create_embedding_client()`: 创建Embedding客户端

### 3. 搜索模块 (`agent/search/`)

- **工具函数**: `search_utils.py`
  - 搜索相关工具函数

- **工具函数**: `web_search.py`
  - 网络搜索功能
  - `create_web_search_tool()`: 创建网络搜索工具

注意：搜索模块不再使用配置文件，通过函数参数直接控制行为。

### 4. 全局RAG配置 (`/`)

- **配置文件**: `rag_config.yaml`
  - LLM配置
  - Agent配置
  - 数据字段配置
  - 多轮对话RAG配置

- **配置类**: `rag_config.py`
  - `LLMConfig`: LLM配置
  - `DataConfig`: 数据字段配置
  - `AgentConfig`: Agent配置（包含kgraph子配置）
  - `MultiDialogueRagConfig`: 多轮对话RAG配置
  - `AppConfig`: 应用主配置（聚合所有子模块配置）

- **加载器**: `rag_loader.py`
  - `RAGConfigLoader`: 加载全局RAG配置，支持动态修改

## 使用示例

### 加载知识图谱配置

```python
from agent.kgraph.kg_loader import KGraphConfigLoader

loader = KGraphConfigLoader()
neo4j_config = loader.neo4j
llm_config = loader.llm
kgraph_agent_config = loader.kgraph_agent
```

### 加载向量检索配置

```python
from agent.milvus.embed_loader import EmbedConfigLoader

loader = EmbedConfigLoader()
milvus_config = loader.milvus
embedding_config = loader.embedding
```

### 使用网络搜索功能

```python
from search.web_search import create_web_search_tool

# 创建网络搜索工具
network_search_tool, network_search_llm, network_tool_node = create_web_search_tool(
    search_cnt=10,          # 网络搜索返回结果数量
    power_model=llm         # LLM实例
)
```

### 加载全局RAG配置

```python
from rag_loader import RAGConfigLoader

loader = RAGConfigLoader()
app_config = loader.config  # AppConfig对象
```

## 配置依赖关系

- `rag_config.py` 依赖于部分子模块的配置类（知识图谱等）
- 各子模块配置相互独立，不依赖其他模块
- 全局RAG配置聚合相关子模块配置
- 搜索模块不使用配置文件，通过函数参数控制行为

## 环境变量支持

所有加载器都支持环境变量替换，使用 `${VAR_NAME}` 格式：

```yaml
neo4j:
  password: ${NEO4J_PASSWORD}
```

## 原配置文件

- `multi_dialogue_agent.yaml` - 原始配置文件（已拆分，可保留或删除）
