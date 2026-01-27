# 说明
    1.260130  文本RAG标准产品  预计完工
    2.核心功能
        1) query强化可以配置
        2) 召回强化可以配置
        3) 过滤可配置
        4) 排序可配置
        5)支持
            向量检索，图谱检索，搜索引擎
    3.核心目的
        标准的RAG体系架构
        配置即可适配业务
        通过配置快速解决业务问题，不同的强化优化，解决不同问题，根据业务需要打开（非必要不要打开，影响性能）


# 参考代码

[medical-rag](https://github.com/ImprintLab/Medical-Graph-RAG)

[open-graph-rag](https://github.com/OpenGraphRAG/MedicalGraphRAG?tab=readme-ov-file)

[medical-graph-rag](https://github.com/OpenGraphRAG/MedicalGraphRAG)

[ai-medical-assistant](https://github.com/zhttyy520/ai-medical-assistant)

# 快速开始

## 创建环境

    1.创建环境
        conda create -name medicalrag python=3.11
    2.安装依赖[只安装依赖]
        pip install -r requirements.txt

## 基础服务

    1.安装docker desktop
        下载地址：  https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
        安装测试：  docker --version (powershell)
    2.安装ollma
        下载地址：  https://ollama.cc/download.html
    3.milvus安装与启动 powershell
        单例配置下载：
        $ curl -O https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml
        $ docker compose up -d    # 首次会拉镜像，耐心等 1–3 分钟
        $ docker ps     # 查看运行容器
    4.neo4j安装与启动 powershell
        $ PS C:\Users\xiang> docker run -d  --name neo4j  -p 7474:7474  -p 7687:7687   -e NEO4J_AUTH=neo4j/12345678 neo4j:latest
        注意：
            用户名必须是 neo4j
            密码长度至少 8位
    5.ollama安装模型 powershell
        $ ollama serve  # 安装并启动 Ollama
        $ ollama pull bge-m3:latest      # 嵌入模型
        $ ollama pull qwen3:4b           # 对话模型，本机测试用一个小号大模型就可以了，真正上线可以采用微调后的领域大模型
        $ ollama pull dengcao/bge-reranker-v2-m3


## 项目结构

### 知识图谱模块 (4.知识图谱/)

#### 核心模块
- **配置管理**
  - `config.py` - 配置参数加载与管理
  - `neo4j_connection.py` - Neo4j数据库连接管理

- **服务层**
  - `embedding_service.py` - 嵌入向量服务，调用大模型API生成嵌入
  - `llm_service.py` - 大模型服务，处理实体关系提取和问答生成

- **数据操作层**
  - `neo4j_operations.py` - 基础CRUD操作（创建实体、关系、更新、删除等）
  - `neo4j_query.py` - 图查询功能（关键字搜索、最短路径、子图查询、三元组查询）
  - `neo4j_save.py` - 批量保存实体和关系

- **检索层**
  - `vector_search.py` - 向量检索，基于嵌入向量的相似度检索
  - `rag_system.py` - RAG系统整合，整合向量检索和知识图谱生成答案

- **工具层**
  - `text_processor.py` - 文本处理，从文件读取和预处理文本

#### 数据流程

**知识构建流程**
```
文本文件 → TextProcessor → 提取文本
                              ↓
                    LLMService → 提取实体和关系
                              ↓
                    EmbeddingService → 生成嵌入向量
                              ↓
                    Neo4jSave → 保存到数据库
```

**RAG查询流程**
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

## 研发说明

    使用声明： 随便学习下载和商业化
    正式版本： 当前还在研发中，可以稍微等待1~2周
             将会是一个产品化的，可快速配置的RAG框架，足够覆盖RAG主流功能

## 研发步骤

    1.数据处理框架    90%
        完善基础算法
    2.词库生成       90%
        停用词缺失
    3.索引构建       90%
        配置无用字段清除
    4.单轮对话RAG   100%
        确认检索字段配置起效
        测试排序字段配置是否生效 weighted
        配置无用字段清除
    5.多轮对话RAG   100%
        召回RRF缺失？
    6.单轮对话agent 100%
        rag结果，经过judge判断，没有不能解决问题，再次转到rag节点进行执行是没有意义的
        建议：  如果判断不能解决问题，要想法扩大检索，比如利用第三方搜索，该搜索要可配置
    7.多轮对话agent 100%
        多个子问题的回答，应该通过大模型进行统一生成一个回答
        当前的检索和数据库查询是二选一，应该修改为并行查询
    8.文本RAG标准产品 60%
        全流程调试通过 100%
        去除冗余代码   100%
        简化代码提升性能
        规范化项目
        完善功能：  支持ollama reranker,embedding,llm
    8.多模态代理模式
    9.修改类的命名
        形成标准化的RAG架构

## 优化方向

### 多轮对话Agent优化
1. 并行检索
   - db检索
   - 搜索
   - 图谱
   - 自研搜索

2. Query增强
   - 改写

3. 召回增强
   - 多路召回
   - 问题分析
   - 上位优化
   - 假设性文档

4. 排序过滤
   - RRF -> 压缩过滤（低相关文档过滤->无关内容过滤->冗余过滤）->重排序ReRanker->重排序(头尾位置)

5. 上面的每一个功能可以配置是否启用

### 数据处理优化
1. 完善数据处理流程
2. 摘要生成
3. 假设性问题生成
4. 元数据生成

### 多模态
1. 基于多轮对话agent改造

## 流程优化

1. 要有判断，是否需要触发RAG，常规问题就不需要走RAG
2. 代码设计原则：
   - 模块化设计，各模块职责清晰
   - 支持缓存机制，提高性能
   - 完善的错误处理
   - 详细的日志输出


# 计算 hash_id（基于 chunk 内容的 MD5）
hash_id = hashlib.md5(chunk.encode('utf-8')).hexdigest()
# 构建 pk = hash_id前16位_chunk_id
pk = f"{hash_id[:16]}_{chunk_id}"

