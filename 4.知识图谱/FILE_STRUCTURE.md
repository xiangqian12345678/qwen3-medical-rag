# 文件结构说明

## 代码文件清单

### 1. 配置文件 (2个)
- **config.json** - 配置文件，包含Neo4j连接、API密钥等配置
- **kg_schema.json** - 知识图谱模式配置，定义实体类型和关系类型

### 2. 核心模块文件 (11个)

#### 连接与配置
- **config.py** (130行) - 配置管理模块，负责加载和管理配置参数
- **neo4j_connection.py** (150行) - Neo4j数据库连接管理模块

#### 服务层
- **embedding_service.py** (200行) - 嵌入向量服务，调用大模型API生成嵌入
- **llm_service.py** (280行) - 大模型服务，处理实体关系提取和问答生成

#### 数据操作层
- **neo4j_operations.py** (300行) - 基础数据库操作（CRUD）
- **neo4j_query.py** (250行) - 图查询功能（最短路径、子图查询等）
- **neo4j_save.py** (200行) - 数据保存功能，批量保存实体和关系

#### 检索层
- **vector_search.py** (280行) - 向量检索功能，基于嵌入向量的相似度检索
- **rag_system.py** (200行) - RAG系统整合，整合向量检索和知识图谱

#### 工具层
- **text_processor.py** (180行) - 文本处理功能，从文件读取和预处理文本

### 3. 主程序文件 (1个)
- **main.py** (300行) - 主程序入口，包含6个完整示例

### 4. 测试和演示文件 (2个)
- **test_basic.py** (150行) - 基础功能测试
- **demo_quick_start.py** (200行) - 快速开始演示

### 5. 配置和文档 (3个)
- **requirements.txt** - Python依赖包列表
- **README.md** - 完整的使用文档
- **FILE_STRUCTURE.md** - 本文件，文件结构说明

## 功能模块说明

### 模块1: 配置管理 (config.py)
- **功能**: 加载和管理配置参数
- **主要类**: Config
- **主要方法**:
  - `load_config()` - 从文件加载配置
  - `get(key, default)` - 获取配置项

### 模块2: 数据库连接 (neo4j_connection.py)
- **功能**: 建立和维护Neo4j数据库连接
- **主要类**: Neo4jConnection
- **主要方法**:
  - `connect()` - 连接数据库
  - `get_driver()` - 获取驱动对象
  - `check_connection()` - 检查连接状态
  - `close()` - 关闭连接

### 模块3: 嵌入服务 (embedding_service.py)
- **功能**: 生成文本的嵌入向量
- **主要类**: EmbeddingService
- **主要方法**:
  - `generate_embedding(text)` - 生成单个文本的嵌入
  - `generate_embeddings_batch(texts)` - 批量生成嵌入
  - `clear_cache()` - 清空缓存

### 模块4: 大模型服务 (llm_service.py)
- **功能**: 调用大模型进行实体关系提取和问答
- **主要类**: LLMService
- **主要方法**:
  - `extract_entities_relations(text, entity_types, relation_types)` - 提取实体关系
  - `generate_answer(question, context)` - 生成答案
  - `generate_rag_answer(question, kg_results, vdb_results)` - 生成RAG答案

### 模块5: 数据库操作 (neo4j_operations.py)
- **功能**: 基础的CRUD操作
- **主要类**: Neo4jOperations
- **主要方法**:
  - `create_entity(name, type, properties)` - 创建实体
  - `create_relationship(source_id, target_id, type, properties)` - 创建关系
  - `get_entity_by_id(entity_id)` - 根据ID获取实体
  - `get_entity_by_name(name)` - 根据名称获取实体
  - `update_entity(entity_id, properties)` - 更新实体
  - `delete_entity(entity_id)` - 删除实体
  - `get_statistics()` - 获取统计信息

### 模块6: 图查询 (neo4j_query.py)
- **功能**: 高级图查询功能
- **主要类**: Neo4jQuery
- **主要方法**:
  - `search_by_keyword(keyword, limit)` - 关键字搜索
  - `shortest_path(source, target)` - 最短路径查询
  - `query_by_entities(entity_ids, depth)` - 根据实体ID查询子图
  - `query_triples(head, relation, tail)` - 三元组查询
  - `get_all_graph(limit)` - 获取整个图谱

### 模块7: 数据保存 (neo4j_save.py)
- **功能**: 批量保存实体和关系
- **主要类**: Neo4jSave
- **主要方法**:
  - `save_entities_and_relationships(entities, relationships)` - 批量保存
  - `save_text_knowledge(text, llm_service, entity_types, relation_types)` - 从文本提取并保存

### 模块8: 向量检索 (vector_search.py)
- **功能**: 基于嵌入向量的相似度检索
- **主要类**: VectorSearch
- **主要方法**:
  - `load_embeddings_from_db()` - 从数据库加载嵌入向量
  - `search_similar_entities(query_text, threshold, top_k)` - 搜索相似实体
  - `search_similar_relationships(query_text, threshold, top_k)` - 搜索相似关系

### 模块9: RAG系统 (rag_system.py)
- **功能**: 整合向量检索和知识图谱，生成最终答案
- **主要类**: RAGSystem
- **主要方法**:
  - `process_query(query_text, ...)` - 处理用户查询
  - `simple_query(query_text)` - 简单查询接口
  - `get_graph_data(limit)` - 获取图谱数据
  - `search_graph(keyword)` - 搜索图谱

### 模块10: 文本处理 (text_processor.py)
- **功能**: 从文件读取和预处理文本
- **主要类**: TextProcessor
- **主要方法**:
  - `load_text_from_file(file_path)` - 从文件加载文本
  - `load_text_from_directory(directory)` - 从目录加载文本
  - `split_text(text, chunk_size, overlap)` - 分割文本
  - `load_and_split_file(file_path, chunk_size, overlap)` - 加载并分割文件

## 数据流程

### 知识构建流程
```
文本文件 → TextProcessor → 提取文本
                              ↓
                    LLMService → 提取实体和关系
                              ↓
                    EmbeddingService → 生成嵌入向量
                              ↓
                    Neo4jSave → 保存到数据库
```

### RAG查询流程
```
用户问题 → RAGSystem
                ↓
    LLMService → 提取查询中的实体
                ↓
    EmbeddingService → 生成查询嵌入
                ↓
    VectorSearch → 向量相似度检索
                ↓
    Neo4jQuery → 图谱查询
                ↓
    LLMService → 生成最终答案
                ↓
    返回结果
```

## 使用流程

### 1. 首次使用 - 构建知识库
```bash
# 步骤1: 安装依赖
pip install -r requirements.txt

# 步骤2: 配置数据库
编辑 config.json 设置Neo4j连接信息和API密钥

# 步骤3: 测试连接
python test_basic.py

# 步骤4: 运行示例
python main.py
```

### 2. 日常使用 - RAG问答
```bash
# 运行快速开始演示
python demo_quick_start.py
```

## 注意事项

1. 所有代码文件都遵循以下规范：
   - 每个文件不超过800行代码
   - 每个函数不超过80行代码
   - 包含完整的文档字符串
   - 包含输入输出示例

2. 代码特点：
   - 模块化设计，各模块职责清晰
   - 支持缓存机制，提高性能
   - 完善的错误处理
   - 详细的日志输出

3. 功能特点：
   - 支持从文本自动提取实体和关系
   - 支持向量相似度检索
   - 支持多种图查询方式
   - 整合RAG生成准确答案
