# 基础RAG系统

医疗知识库检索增强生成(RAG)系统，支持混合检索和向量融合。

## 目录结构

```
5.基础RAG/
├── config/              # 配置模块
│   ├── __init__.py
│   ├── loader.py        # 配置加载器
│   └── models.py        # 配置数据模型
├── embed/               # 嵌入模块
│   ├── __init__.py
│   ├── vocab.py         # 词表管理
│   └── bm25.py          # BM25稀疏向量
├── __init__.py          # 模块导出
├── knowledge_base.py    # 知识库核心类
├── retriever.py         # LangChain检索器
├── rag.py               # RAG系统
├── utils.py             # 工具函数
├── prompts.py           # Prompt模板
├── simple_rag.yaml      # 配置文件
└── example.py           # 使用示例
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r ../../environment.yml
```

### 2. 配置设置

编辑 `simple_rag.yaml` 配置文件，设置Milvus连接、嵌入模型和LLM配置。

### 3. 运行示例

```python
from rag import SimpleRAG
from config.loader import ConfigLoader

# 加载配置
config_loader = ConfigLoader()
rag = SimpleRAG(config=config_loader.config)

# 执行查询
result = rag.generate_answer("肚子痛该怎么办？", return_document=True)
print(result['answer'])
```

### 4. 命令行运行

```bash
python example.py
```

## 模块说明

### config - 配置模块

- **ConfigLoader**: 配置加载器，支持动态修改配置
- **模型定义**: MilvusConfig, DenseFieldConfig, SparseFieldConfig, LLMConfig等

### embed - 嵌入模块

- **Vocabulary**: 词表管理，支持BM25分词和IDF计算
- **BM25Vectorizer**: BM25稀疏向量化器
- **BM25SparseEmbedding**: LangChain接口的稀疏向量嵌入

### knowledge_base - 知识库

- **MedicalKnowledgeBase**: 核心知识库类，支持单路检索和混合检索

### retriever - 检索器

- **MedicalRetriever**: LangChain标准接口的检索器

### rag - RAG系统

- **SimpleRAG**: 基础RAG系统，提供问题回答功能

## 配置说明

### Milvus配置

```yaml
milvus:
  uri: http://localhost:19530      # Milvus地址
  token: null                      # 认证令牌
  collection_name: medical_knowledge  # 集合名称
  drop_old: false                  # 是否删除旧集合
  auto_id: false                   # 是否自动生成ID
```

### 稠密向量字段配置

```yaml
dense_fields:
  chunk:                          # 文本块稠密向量
    provider: ollama
    model: bge-m3:latest
    dimension: 1024
  parent_chunk:                   # 父级文本块稠密向量
    provider: ollama
    model: bge-m3:latest
    dimension: 1024
  questions:                      # 问题稠密向量
    provider: ollama
    model: bge-m3:latest
    dimension: 1024
```

### 稀疏向量字段配置

```yaml
sparse_fields:
  chunk:                          # 文本块稀疏向量(BM25)
    provider: self
    vocab_path_or_name: ../output/vocab/vocab.pkl.gz
    k1: 1.5
    b: 0.75
```

### RAG检索配置

```yaml
rag:
  default_fields:                   # 默认检索字段
    - anns_field: chunk_dense
      metric_type: COSINE
      search_params: {"ef": 64}
      limit: 50
    - anns_field: parent_chunk_dense
      metric_type: COSINE
      search_params: {"ef": 64}
      limit: 50
    - anns_field: questions_dense
      metric_type: COSINE
      search_params: {"ef": 64}
      limit: 50
    - anns_field: chunk_sparse
      metric_type: IP
      search_params: {"drop_ratio_search": 0.0}
      limit: 50
  fusion:                           # 向量融合
    method: rrf
    k: 60
    weights:
      chunk_dense: 0.35
      parent_chunk_dense: 0.35
      questions_dense: 0.20
      chunk_sparse: 0.10
  limit: 5                          # 最终返回数量
  top_k: 50                         # 融合前各路检索的返回数量
```

## API文档

### SimpleRAG

```python
class SimpleRAG:
    def __init__(self, config=None, config_path=None, search_config=None):
        """初始化RAG系统"""

    def answer(self, query: str, return_document: bool = False) -> dict:
        """回答问题
        返回: {
            "answer": str,              # LLM答案
            "search_time": float,       # 检索耗时
            "generation_time": float,   # 生成耗时
            "documents": List[Document]  # 检索文档(可选)
        }
        """

    def batch_answer(self, queries: List[str], return_document: bool = False) -> List[dict]:
        """批量回答问题"""

    def update_search_config(self, search_config: SearchRequest):
        """更新检索配置"""
```

### MedicalKnowledgeBase

```python
class MedicalKnowledgeBase:
    def __init__(self, app_config: AppConfig):
        """初始化知识库"""

    def search(self, req: SearchRequest) -> List[Document]:
        """执行检索，返回Document列表"""
```

## 注意事项

1. 确保 Milvus 服务已启动
2. 确保 词表文件已正确生成
3. 确保 嵌入模型和LLM服务可用
