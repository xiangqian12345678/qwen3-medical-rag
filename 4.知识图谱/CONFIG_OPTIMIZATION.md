# 知识图谱模块配置优化说明

## 优化内容

### 1. 配置文件格式改进

**改进前：**
- 使用 `config/config.json` 单一配置文件
- Neo4j 和大模型配置混在一起
- 配置项使用大写命名风格（NEO4J_URI、DASHSCOPE_API_KEY 等）

**改进后：**
- 使用 `config/kg_config.yaml` YAML 格式配置文件
- 按功能模块区分配置区块（neo4j、llm、embedding、knowledge_graph、rag）
- 配置项使用小写加下划线命名风格，更符合 Python 最佳实践

### 2. 配置文件结构

```yaml
neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: "12345678"
  database: neo4j
  max_connection_lifetime: 3600
  max_connection_pool_size: 50
  connection_timeout: 30.0
  acquisition_timeout: 60.0

llm:
  provider: dashscope
  api_key: sk-9de21cd0776947ec9ef2396f7110cb6d
  api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
  model: qwen-plus
  temperature: 0.3
  max_tokens: 2048

embedding:
  provider: dashscope
  url: https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
  model: text-embedding-v2
  dimension: 1536

knowledge_graph:
  schema_file: config/kg_schema.json
  vector_index_path: knowledge_index/kg_vector_index.pkl
  chunk_size: 1000
  chunk_overlap: 100
  graph_top_k: 10
  similarity_threshold: 0.7

rag:
  depth: 2
  top_k: 5
  max_tokens: 1024
  cache_enabled: true
  console_debug: true
```

### 3. 配置管理类改进

**新增配置类：**
- `Neo4jConfig`: Neo4j 数据库专用配置
- `LLMConfig`: 大模型服务专用配置
- `EmbeddingConfig`: 嵌入服务专用配置
- `KGConfig`: 知识图谱专用配置
- `RAGConfig`: RAG 系统专用配置

**改进后的特性：**
1. 每个配置类只负责自己的功能模块
2. 使用 `@property` 装饰器提供便捷的属性访问方式
3. 支持点号分隔的配置路径访问（如 `config.get("neo4j.uri")`）
4. 保持向后兼容，旧的属性访问方式仍然有效

### 4. 代码修改

#### 修改的文件：

1. **config.py**
   - 改用 `yaml.safe_load()` 加载 YAML 配置
   - 新增多个配置类（Neo4jConfig、LLMConfig 等）
   - 保留向后兼容的 `__getattr__` 方法

2. **neo4j_connection.py**
   - 改用 `neo4j_config` 替代 `config`
   - 使用配置类的属性访问方式

3. **llm_service.py**
   - 改用 `llm_config` 替代 `config`
   - 使用配置类的属性访问方式

4. **embedding_service.py**
   - 改用 `embedding_config` 替代 `config`
   - 使用配置类的属性访问方式

5. **config/kg_config.yaml** (新建)
   - YAML 格式的配置文件

6. **config/config.json** (已删除)
   - 删除旧的 JSON 格式配置文件

### 5. 使用示例

```python
from config import config, neo4j_config, llm_config, embedding_config, kg_config, rag_config

# 方式1: 使用全局配置对象（向后兼容）
uri = config.NEO4J_URI  # 仍然支持
api_key = config.DASHSCOPE_API_KEY

# 方式2: 使用功能配置类
uri = neo4j_config.uri
user = neo4j_config.user
password = neo4j_config.password

# 方式3: 使用点号分隔的路径
uri = config.get("neo4j.uri")
user = config.get("neo4j.user")

# 方式4: 获取整个配置区块
neo4j_settings = config.get_section("neo4j")
```

### 6. 优势

1. **模块化**: 不同功能的配置分离，便于管理和维护
2. **可读性**: YAML 格式比 JSON 更易读，支持注释
3. **类型安全**: 配置类提供类型化的属性访问
4. **向后兼容**: 保留了旧的配置访问方式，不破坏现有代码
5. **可扩展性**: 新增功能只需添加新的配置区块和配置类
6. **一致性**: 与项目其他模块（如多轮对话Agent）的配置风格保持一致

### 7. 依赖要求

- `pyyaml>=6.0.2` (已在 requirements.txt 中)
- 无需额外安装其他依赖
