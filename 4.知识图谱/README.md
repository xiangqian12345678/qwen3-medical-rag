# 基于知识图谱的RAG系统

完整的基于Neo4j知识图谱的检索增强生成(RAG)系统，包含单轮会话示例。

## 功能特性

1. **Neo4j数据库操作**：完整的CRUD操作支持
2. **大模型集成**：支持通义千问等大模型进行实体关系提取和问答
3. **向量检索**：基于嵌入向量的相似度检索
4. **图查询**：最短路径、子图查询等高级图查询功能
5. **RAG整合**：整合向量检索和知识图谱检索生成最终答案

## 文件结构

```
tmp/
├── config.json           # 配置文件
├── config.py            # 配置管理模块
├── neo4j_connection.py  # Neo4j连接管理 (150行)
├── embedding_service.py # 嵌入向量服务 (200行)
├── llm_service.py       # 大模型服务 (280行)
├── neo4j_operations.py # 基础数据库操作 (300行)
├── neo4j_query.py       # 图查询功能 (250行)
├── neo4j_save.py        # 数据保存功能 (200行)
├── vector_search.py     # 向量检索功能 (280行)
├── rag_system.py        # RAG系统整合 (200行)
├── text_processor.py    # 文本处理功能 (180行)
├── main.py              # 主程序和示例 (300行)
├── kg_schema.json       # 知识图谱模式配置
└── requirements.txt     # 依赖包列表
```

## 快速开始

### 1.创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS

``` 

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置数据库

编辑 `config.json`，设置Neo4j数据库连接信息：

```json
{
  "NEO4J_URI": "bolt://localhost:7687",
  "NEO4J_USER": "neo4j",
  "NEO4J_PASSWORD": "your_password",
  "TONGYI_KEY": "your_api_key"
}
```

### 4. 运行示例

```bash
python main.py
```

## 使用示例

### 示例1: 基础数据库操作

```python
from neo4j_connection import Neo4jConnection
from neo4j_operations import Neo4jOperations

# 连接数据库
conn = Neo4jConnection()
conn.connect()

# 创建操作对象
ops = Neo4jOperations(conn)

# 创建实体
entity_id = ops.create_entity(
    name="阿司匹林",
    entity_type="药物",
    properties={"成分": "乙酰水杨酸"}
)

# 创建关系
symptom_id = ops.create_entity("头痛", "症状", {})
rel_id = ops.create_relationship(
    source_id=entity_id,
    target_id=symptom_id,
    rel_type="治疗"
)

conn.close()
```

### 示例2: 从文本提取并保存知识

```python
from neo4j_connection import Neo4jConnection
from neo4j_save import Neo4jSave
from llm_service import LLMService
from embedding_service import EmbeddingService

# 初始化服务
conn = Neo4jConnection()
conn.connect()

embed_service = EmbeddingService()
llm_service = LLMService()
saver = Neo4jSave(conn, embed_service)

# 从文本提取并保存
text = "阿司匹林可以治疗头痛和发热。"
saver.save_text_knowledge(
    text,
    llm_service,
    entity_types=["药物", "症状", "疾病"],
    relation_types=["治疗", "导致"]
)

# 关闭服务
embed_service.close()
llm_service.close()
conn.close()
```

### 示例3: RAG查询

```python
from neo4j_connection import Neo4jConnection
from rag_system import RAGSystem

# 初始化RAG系统
conn = Neo4jConnection()
conn.connect()

rag = RAGSystem(conn)

# 处理查询
result = rag.process_query(
    query_text="阿司匹林可以治疗什么？",
    entity_types=["药物", "症状", "疾病"],
    relation_types=["治疗", "导致"],
    depth=2,
    similarity_threshold=0.7,
    top_k=5
)

# 获取答案
answer = result['answer']
print(answer)

rag.close()
conn.close()
```

### 示例4: 完整的RAG单轮会话

```python
from rag_system import RAGSystem
from neo4j_connection import Neo4jConnection

# 初始化系统
conn = Neo4jConnection()
conn.connect()
rag = RAGSystem(conn)

# 用户会话
while True:
    user_input = input("请输入您的问题: ")

    if user_input.lower() in ['quit', 'exit']:
        break

    # 处理查询
    result = rag.process_query(user_input)

    # 显示答案
    print(f"回答: {result['answer']}")
    print(f"处理时间: {result['processing_time']:.2f}s")

rag.close()
conn.close()
```

## 配置说明

### config.json 配置项

| 配置项                  | 说明         | 默认值                   |
|----------------------|------------|-----------------------|
| NEO4J_URI            | Neo4j数据库地址 | bolt://localhost:7687 |
| NEO4J_USER           | Neo4j用户名   | neo4j                 |
| NEO4J_PASSWORD       | Neo4j密码    | 12345678              |
| TONGYI_KEY           | 通义千问API密钥  | (需要配置)                |
| EMBEDDING_URL        | 嵌入模型API地址  | (默认值)                 |
| EMBEDDING_MODEL      | 嵌入模型名称     | text-embedding-v1     |
| GRAPH_TOP_K          | 图查询返回数量    | 10                    |
| SIMILARITY_THRESHOLD | 相似度阈值      | 0.7                   |

## 知识图谱模式

系统默认支持以下实体类型和关系类型：

### 实体类型

- **药物**: 成分、剂量、适应症、副作用等属性
- **症状**: 名称、描述、严重程度等属性
- **疾病**: 病因、治疗方法、症状等属性

### 关系类型

- **治疗**: 药物 → 症状/疾病
- **导致**: 疾病 → 症状
- **属于**: 症状 → 疾病

可以通过修改 `kg_schema.json` 来自定义知识图谱模式。

## API文档

### Neo4jOperations 类

基础数据库操作类。

| 方法                    | 说明       | 输入                                     | 输出          |
|-----------------------|----------|----------------------------------------|-------------|
| create_entity()       | 创建实体     | name, type, properties                 | entity_id   |
| create_relationship() | 创建关系     | source_id, target_id, type, properties | rel_id      |
| get_entity_by_id()    | 根据ID获取实体 | entity_id                              | entity_dict |
| get_entity_by_name()  | 根据名称获取实体 | name                                   | entity_dict |
| update_entity()       | 更新实体     | entity_id, properties                  | bool        |
| delete_entity()       | 删除实体     | entity_id                              | bool        |
| get_statistics()      | 获取统计信息   | 无                                      | stats_dict  |

### Neo4jQuery 类

图查询类。

| 方法                  | 说明         | 输入                   | 输出           |
|---------------------|------------|----------------------|--------------|
| search_by_keyword() | 关键字搜索      | keyword, limit       | graph_dict   |
| shortest_path()     | 最短路径查询     | source, target       | paths_list   |
| query_by_entities() | 根据实体ID查询子图 | entity_ids, depth    | records_list |
| query_triples()     | 三元组查询      | head, relation, tail | graph_dict   |
| get_all_graph()     | 获取整个图谱     | limit                | graph_dict   |

### RAGSystem 类

RAG系统整合类。

| 方法               | 说明     | 输入              | 输出          |
|------------------|--------|-----------------|-------------|
| process_query()  | 处理用户查询 | query_text, ... | result_dict |
| simple_query()   | 简单查询接口 | query_text      | answer_str  |
| get_graph_data() | 获取图谱数据 | limit           | graph_dict  |
| search_graph()   | 搜索图谱   | keyword         | graph_dict  |

## 依赖要求

- Python >= 3.8
- Neo4j >= 5.0
- 通义千问API密钥

## 注意事项

1. 确保Neo4j数据库已启动并可访问
2. 需要配置有效的通义千问API密钥
3. 首次运行时会加载所有嵌入向量到内存
4. 大规模知识库建议增加内存配置

## 许可证

本项目代码仅供学习参考使用。
