# 基于知识图谱的RAG系统

完整的基于Neo4j知识图谱的检索增强生成(RAG)系统，支持单轮会话，整合向量检索和知识图谱检索生成高质量答案。

## 项目概述

本系统提供了一个完整的医疗领域知识图谱RAG解决方案，主要特点：

- **完整的Neo4j操作**：实体和关系的CRUD操作
- **大模型集成**：使用通义千问进行实体关系提取和答案生成
- **向量检索**：基于嵌入向量的相似度检索
- **图查询**：关键字搜索、最短路径、子图查询等高级功能
- **RAG整合**：结合知识图谱和向量检索生成准确答案
- **配置化管理**：统一的配置管理和知识图谱Schema定义
- **模块化配置**：使用YAML格式，按功能模块组织配置

## 功能特性

### ✅ 已实现功能

1. **完整的Neo4j操作**
   - 数据库连接管理
   - 实体CRUD操作（创建、查询、更新、删除）
   - 关系CRUD操作
   - 嵌入向量存储
   - 统计信息查询

2. **从文本读取数据**
   - 支持多种文件格式（txt, pdf, docx, md）
   - 文本分割和预处理
   - 批量文件处理

3. **大模型调用生成元组**
   - 实体关系提取
   - 支持自定义实体类型和关系类型
   - JSON格式输出
   - 错误处理和重试机制

4. **实体和关系存储到Neo4j**
   - 批量保存功能
   - 自动生成嵌入向量
   - 避免重复创建（MERGE操作）
   - 缓存机制优化性能

5. **数据查询**
   - 关键字搜索
   - 最短路径查询
   - 子图查询
   - 三元组查询
   - 全图查询

6. **根据用户问题进行Neo4j查询**
   - 从问题中提取实体
   - 向量相似度检索
   - 图谱多跳查询
   - 结果整合

7. **生成RAG结果**
   - 整合知识图谱和向量检索结果
   - 调用大模型生成答案
   - 支持多种查询模式
   - 完整的错误处理

## 文件结构

```
4.知识图谱/
├── config/                        # 配置文件目录
│   ├── kg_config.yaml             # 知识图谱配置文件（YAML格式）
│   └── kg_schema.json            # 知识图谱模式（实体类型和关系类型）
├── config.py                     # 配置管理模块
├── neo4j_connection.py          # Neo4j数据库连接管理
├── embedding_service.py         # 嵌入向量生成服务
├── llm_service.py               # 大模型服务
├── neo4j_operations.py          # 基础数据库CRUD操作
├── neo4j_query.py               # 图查询功能
├── neo4j_save.py                # 数据保存功能
├── vector_search.py             # 向量相似度检索
├── rag_system.py                # RAG系统整合
├── text_processor.py            # 文本处理功能
├── main.py                      # 主程序和示例
├── demo_quick_start.py          # 快速开始演示
├── test_basic.py                # 基础功能测试
└── README.md                    # 本文档
```

### 核心模块说明

| 模块 | 功能 | 代码行数 |
|-----|------|---------|
| config.py | 配置管理，支持统一配置和Schema管理 | ~220行 |
| neo4j_connection.py | Neo4j连接管理，连接池配置 | ~150行 |
| embedding_service.py | 嵌入向量生成，支持DashScope API | ~200行 |
| llm_service.py | 大模型调用，实体提取和答案生成 | ~280行 |
| neo4j_operations.py | 基础CRUD操作 | ~300行 |
| neo4j_query.py | 图查询功能 | ~420行 |
| neo4j_save.py | 数据保存和批量处理 | ~200行 |
| vector_search.py | 向量相似度检索 | ~280行 |
| rag_system.py | RAG系统整合 | ~200行 |
| text_processor.py | 文本处理 | ~180行 |
| main.py | 主程序和6个完整示例 | ~360行 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据库

编辑 `config/kg_config.yaml`，设置Neo4j数据库连接信息和API密钥：

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
  api_key: your-api-key-here
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

配置项说明：

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| neo4j.uri | Neo4j数据库地址 | bolt://localhost:7687 |
| neo4j.user | Neo4j用户名 | neo4j |
| neo4j.password | Neo4j密码 | 12345678 |
| llm.api_key | 通义千问API密钥 | (需配置) |
| embedding.model | 嵌入模型名称 | text-embedding-v2 |
| knowledge_graph.graph_top_k | 图查询返回数量 | 10 |
| knowledge_graph.similarity_threshold | 相似度阈值 | 0.7 |

### 3. 配置知识图谱Schema

编辑 `config/kg_schema.json`，定义实体类型和关系类型：

```json
{
  "name": "医疗知识图谱",
  "description": "医疗领域知识图谱，支持药物、症状、疾病等实体类型",
  "entity_types": [
    {
      "name": "药物",
      "properties": ["名称", "成分", "剂量", "适应症", "副作用"]
    },
    {
      "name": "症状",
      "properties": ["名称", "描述", "严重程度"]
    },
    {
      "name": "疾病",
      "properties": ["名称", "病因", "治疗方法", "症状"]
    }
  ],
  "relationship_types": [
    {
      "name": "治疗",
      "source": ["药物"],
      "target": ["症状", "疾病"]
    },
    {
      "name": "导致",
      "source": ["疾病"],
      "target": ["症状"]
    },
    {
      "name": "属于",
      "source": ["症状"],
      "target": ["疾病"]
    }
  ]
}
```

### 4. 运行示例

```bash
# 测试基础功能
python test_basic.py

# 运行完整示例
python main.py

# 快速开始演示
python demo_quick_start.py
```

## 配置使用说明

### 配置类说明

系统提供了多个专用的配置类，每个类负责特定功能模块的配置：

```python
from config import (
    config,              # 全局配置对象（向后兼容）
    neo4j_config,        # Neo4j配置
    llm_config,          # 大模型配置
    embedding_config,    # 嵌入服务配置
    kg_config,           # 知识图谱配置
    rag_config           # RAG系统配置
)
```

### 配置访问方式

```python
# 方式1: 使用全局配置对象（向后兼容）
uri = config.NEO4J_URI  # 仍然支持旧的大写命名方式
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

### 配置类特性

- **模块化**: 不同功能的配置分离，便于管理和维护
- **可读性**: YAML 格式支持注释，比 JSON 更易读
- **类型安全**: 配置类提供类型化的属性访问
- **向后兼容**: 保留了旧的配置访问方式，不破坏现有代码
- **可扩展性**: 新增功能只需添加新的配置区块和配置类
- **一致性**: 与项目其他模块的配置风格保持一致

## 使用示例

### 示例1: 基础数据库操作

演示如何创建实体和关系：

```python
from neo4j_connection import Neo4jConnection
from neo4j_operations import Neo4jOperations

# 连接数据库
conn = Neo4jConnection()
conn.connect()

# 创建操作对象
ops = Neo4jOperations(conn)

# 创建实体
aspirin_id = ops.create_entity(
    name="阿司匹林",
    entity_type="药物",
    properties={"成分": "乙酰水杨酸"}
)
headache_id = ops.create_entity("头痛", "症状", {})

# 创建关系
ops.create_relationship(aspirin_id, headache_id, "治疗", {})

conn.close()
```

### 示例2: 从文本提取并保存知识

演示使用大模型从文本提取实体关系并保存到数据库：

```python
from neo4j_connection import Neo4jConnection
from neo4j_save import Neo4jSave
from llm_service import LLMService
from embedding_service import EmbeddingService

conn = Neo4jConnection()
conn.connect()

embed_service = EmbeddingService()
llm_service = LLMService()
saver = Neo4jSave(conn, embed_service)

# 自动从kg_schema.json读取实体类型和关系类型
text = "阿司匹林可以治疗头痛和发热。"
saver.save_text_knowledge(text, llm_service)

embed_service.close()
llm_service.close()
conn.close()
```

### 示例3: 查询知识图谱

演示如何查询知识图谱：

```python
from neo4j_connection import Neo4jConnection
from neo4j_query import Neo4jQuery

conn = Neo4jConnection()
conn.connect()
query = Neo4jQuery(conn)

# 关键字搜索
result = query.search_by_keyword("阿司匹林", limit=20)

# 三元组查询
result = query.query_triples(head="阿司匹林", relation="治疗")

# 获取整个图谱
graph = query.get_all_graph(limit=50)

conn.close()
```

### 示例4: RAG查询

演示完整的RAG查询流程：

```python
from neo4j_connection import Neo4jConnection
from rag_system import RAGSystem

conn = Neo4jConnection()
conn.connect()
rag = RAGSystem(conn)

# 自动从kg_schema.json读取实体类型和关系类型
result = rag.process_query(
    query_text="阿司匹林可以治疗什么？",
    depth=2,
    similarity_threshold=0.7,
    top_k=5
)

print(f"答案: {result['answer']}")
print(f"处理时间: {result['processing_time']:.2f}s")

rag.close()
conn.close()
```

### 示例5: 处理文档并构建知识库

演示从文档文件批量处理并构建知识库：

```python
from neo4j_connection import Neo4jConnection
from neo4j_save import Neo4jSave
from llm_service import LLMService
from embedding_service import EmbeddingService
from text_processor import TextProcessor

conn = Neo4jConnection()
conn.connect()

embed_service = EmbeddingService()
llm_service = LLMService()
saver = Neo4jSave(conn, embed_service)
processor = TextProcessor()

# 处理文档目录
documents_dir = "../data/graph"
texts = processor.load_text_from_directory(documents_dir)

# 批量处理文本
for text in texts:
    saver.save_text_knowledge(text, llm_service)

embed_service.close()
llm_service.close()
conn.close()
```

### 示例6: 完整的RAG单轮会话

演示从用户问题到生成答案的完整流程：

```python
from rag_system import RAGSystem
from neo4j_connection import Neo4jConnection

conn = Neo4jConnection()
conn.connect()
rag = RAGSystem(conn)

# 用户会话
while True:
    user_input = input("请输入您的问题: ")

    if user_input.lower() in ['quit', 'exit']:
        break

    result = rag.process_query(user_input)
    print(f"回答: {result['answer']}")
    print(f"处理时间: {result['processing_time']:.2f}s")

rag.close()
conn.close()
```

## API文档

### Config 类

配置管理类，负责加载和管理配置参数。

```python
from config import config

# 配置访问
neo4j_uri = config.get("neo4j.uri")
neo4j_section = config.get_section("neo4j")
```

**Config 类方法:**

| 方法 | 说明 | 返回值 |
|-----|------|--------|
| `get(key, default)` | 获取配置项，支持点号分隔路径 | 配置值 |
| `get_section(section_name)` | 获取整个配置区块 | Dict |

**配置类方法:**

所有配置类（Neo4jConfig、LLMConfig等）都支持通过属性直接访问配置值。

### KGSchema 类

知识图谱Schema管理类。

```python
from config import KGSchema

# 知识图谱Schema
kg_schema = KGSchema("config/kg_schema.json")
entity_types = kg_schema.get_entity_types()
relationship_types = kg_schema.get_relationship_types()
```

**KGSchema 类方法:**

| 方法 | 说明 | 返回值 |
|-----|------|--------|
| `get_entity_types()` | 获取所有实体类型 | List[str] |
| `get_entity_properties(type)` | 获取实体类型的属性 | List[str] |
| `get_relationship_types()` | 获取所有关系类型 | List[str] |
| `get_relationship_info(type)` | 获取关系类型信息 | Dict |
| `format_extraction_prompt(text)` | 格式化提取提示词 | str |

### Neo4jOperations 类

基础数据库操作类。

| 方法 | 说明 | 输入 | 输出 |
|-----|------|------|------|
| `create_entity()` | 创建实体 | name, type, properties | entity_id |
| `create_relationship()` | 创建关系 | source_id, target_id, type, properties | rel_id |
| `get_entity_by_id()` | 根据ID获取实体 | entity_id | entity_dict |
| `get_entity_by_name()` | 根据名称获取实体 | name | entity_dict |
| `update_entity()` | 更新实体 | entity_id, properties | bool |
| `delete_entity()` | 删除实体 | entity_id | bool |
| `get_statistics()` | 获取统计信息 | 无 | stats_dict |

### Neo4jQuery 类

图查询类。

| 方法 | 说明 | 输入 | 输出 |
|-----|------|------|------|
| `search_by_keyword()` | 关键字搜索 | keyword, limit | graph_dict |
| `shortest_path()` | 最短路径查询 | source, target | paths_list |
| `query_by_entities()` | 根据实体ID查询子图 | entity_ids, depth | records_list |
| `query_triples()` | 三元组查询 | head, relation, tail | graph_dict |
| `get_all_graph()` | 获取整个图谱 | limit | graph_dict |

### RAGSystem 类

RAG系统整合类。

| 方法 | 说明 | 输入 | 输出 |
|-----|------|------|------|
| `process_query()` | 处理用户查询 | query_text, depth, threshold, top_k | result_dict |
| `simple_query()` | 简单查询接口 | query_text | answer_str |
| `get_graph_data()` | 获取图谱数据 | limit | graph_dict |
| `search_graph()` | 搜索图谱 | keyword | graph_dict |

### EmbeddingService 类

嵌入向量生成服务。

| 方法 | 说明 | 输入 | 输出 |
|-----|------|------|------|
| `get_embedding()` | 获取文本嵌入向量 | text | List[float] |
| `get_embeddings_batch()` | 批量获取嵌入向量 | texts | List[List[float]] |
| `close()` | 关闭服务 | 无 | None |

### LLMService 类

大模型服务。

| 方法 | 说明 | 输入 | 输出 |
|-----|------|------|------|
| `extract_entities_relations()` | 提取实体关系 | text, entity_types, relation_types | Dict |
| `generate_rag_answer()` | 生成RAG答案 | query, kg_results, vdb_results | str |
| `close()` | 关闭服务 | 无 | None |

## 代码规范

### ✅ 已遵循的规范

1. **文件大小限制**
   - 每个文件不超过800行代码
   - 实际最大文件: neo4j_query.py (约420行)

2. **函数大小限制**
   - 每个函数不超过80行代码
   - 实际最大函数: 约60行

3. **代码注释**
   - 所有函数都有完整的文档字符串
   - 包含功能说明
   - 包含输入输出示例
   - 包含参数说明

4. **模块化设计**
   - 功能模块职责清晰
   - 模块间低耦合
   - 易于维护和扩展

5. **配置管理规范**
   - 使用YAML格式配置文件
   - 按功能模块区分配置区块
   - 配置项使用小写加下划线命名风格
   - 提供专用的配置类

## 依赖要求

```
neo4j>=5.0.0
openai>=1.0.0
httpx>=0.24.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyyaml>=6.0.2
```

## 注意事项

1. **数据库要求**
   - 需要安装并启动Neo4j 5.0+
   - 确保数据库可访问
   - 建议配置足够的内存

2. **API密钥**
   - 需要配置有效的通义千问API密钥
   - 密钥配置在config/kg_config.yaml中

3. **首次运行**
   - 首次运行会从数据库加载所有嵌入向量
   - 可能需要较长时间
   - 建议先用小数据集测试

4. **性能优化**
   - 使用缓存机制减少API调用
   - 支持批量操作
   - 大规模知识库建议增加内存配置

5. **配置管理**
   - 推荐使用新的配置类（neo4j_config、llm_config等）
   - 旧的配置访问方式仍然可用，但建议逐步迁移
   - 配置文件支持注释，便于理解和维护

## 文件统计

- **总文件数**: 19个（含配置文件）
- **代码文件**: 13个
- **配置文件**: 2个（在config目录）
- **文档文件**: 4个
- **总代码行数**: 约3000行
- **平均文件行数**: 约250行
- **最大文件行数**: 420行 (neo4j_query.py)
- **最大函数行数**: 约60行

## 总结

本系统提供了完整的基于Neo4j知识图谱的RAG解决方案，包含：

✅ 完整的neo4j操作代码
✅ 从文本中读取数据的功能
✅ 大模型调用生成元组的功能
✅ 实体和关系存储到neo4j数据库
✅ 完整的数据查询功能
✅ Neo4j使用例子（6个示例）
✅ 根据用户问题进行neo4j查询
✅ 生成RAG结果的例子
✅ 代码功能拆分合理（13个模块）
✅ 每个文件不超过800行
✅ 每个函数不超过80行
✅ 不包含前端展示内容
✅ 所有函数都有输入输出示例和代码注释
✅ 配置文件已整理到config目录
✅ 使用YAML格式配置文件，支持模块化配置
✅ 提供专用配置类，类型安全的配置访问

所有代码都已经过精心设计，可以直接使用。建议先运行test_basic.py测试基础功能，然后运行main.py查看完整示例。
