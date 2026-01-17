# 代码修改总结

## 修改概述
本次修改将4.知识图谱模块的代码重构为从配置文件读取配置和知识图谱Schema信息，实现了配置与代码的分离。

## 修改内容

### 1. config.py
**文件路径**: `config.py`

**主要修改**:
- 修改 `Config` 类，默认从 `config/config.json` 读取配置文件（之前从根目录的 `config.json` 读取）
- 新增 `KGSchema` 类，用于加载和管理知识图谱Schema配置
  - 从 `config/kg_schema.json` 读取知识图谱Schema
  - 提供以下方法：
    - `get_entity_types()`: 获取所有实体类型
    - `get_entity_properties(entity_type)`: 获取指定实体类型的属性
    - `get_relationship_types()`: 获取所有关系类型
    - `get_relationship_info(rel_type)`: 获取指定关系类型的详细信息
    - `get_extraction_prompt()`: 获取实体关系提取提示词
    - `format_extraction_prompt(text)`: 格式化提取提示词
- 创建全局实例 `kg_schema`，供其他模块使用

**修改示例**:
```python
# 之前
config_file = os.path.join(script_dir, "config.json")

# 之后
config_file = os.path.join(script_dir, "config", "config.json")
```

### 2. neo4j_save.py
**文件路径**: `neo4j_save.py`

**主要修改**:
- 导入 `kg_schema` 实例
- 修改 `save_text_knowledge()` 方法，使 `entity_types` 和 `relation_types` 参数变为可选
- 当未指定实体类型和关系类型时，自动从 `kg_schema.json` 读取

**修改示例**:
```python
# 之前
def save_text_knowledge(self, text: str,
                        llm_service,
                        entity_types: List[str],
                        relation_types: List[str]) -> bool:
    result = llm_service.extract_entities_relations(
        text, entity_types, relation_types
    )

# 之后
def save_text_knowledge(self, text: str,
                        llm_service,
                        entity_types: List[str] = None,
                        relation_types: List[str] = None) -> bool:
    if entity_types is None:
        entity_types = kg_schema.get_entity_types()
    if relation_types is None:
        relation_types = kg_schema.get_relationship_types()

    result = llm_service.extract_entities_relations(
        text, entity_types, relation_types
    )
```

### 3. rag_system.py
**文件路径**: `rag_system.py`

**主要修改**:
- 导入 `kg_schema` 实例
- 修改 `process_query()` 方法，使 `entity_types` 和 `relation_types` 参数变为可选
- 当未指定实体类型和关系类型时，自动从 `kg_schema.json` 读取

**修改示例**:
```python
# 之前
def process_query(self, query_text: str,
                  entity_types: List[str] = None,
                  relation_types: List[str] = None,
                  ...):
    entity_types = entity_types or ["药物", "症状", "疾病"]
    relation_types = relation_types or ["治疗", "导致"]

# 之后
def process_query(self, query_text: str,
                  entity_types: List[str] = None,
                  relation_types: List[str] = None,
                  ...):
    entity_types = entity_types or kg_schema.get_entity_types()
    relation_types = relation_types or kg_schema.get_relationship_types()
```

### 4. llm_service.py
**文件路径**: `llm_service.py`

**主要修改**:
- 导入 `kg_schema` 实例
- 修改 `extract_entities_relations()` 方法，使 `entity_types` 和 `relation_types` 参数变为可选
- 当未指定实体类型和关系类型时，自动从 `kg_schema.json` 读取

**修改示例**:
```python
# 之前
def extract_entities_relations(self, text: str, entity_types: List[str],
                               relation_types: List[str]) -> Dict:

# 之后
def extract_entities_relations(self, text: str, entity_types: List[str] = None,
                               relation_types: List[str] = None) -> Dict:
    if entity_types is None:
        entity_types = kg_schema.get_entity_types()
    if relation_types is None:
        relation_types = kg_schema.get_relationship_types()
```

### 5. main.py
**文件路径**: `main.py`

**主要修改**:
- 导入 `kg_schema` 实例
- 移除所有硬编码的 `entity_types` 和 `relation_types`
- 修改以下示例函数：
  - `example_2_extract_and_save()`
  - `example_4_rag_query()`
  - `example_5_process_documents()`
  - `example_6_complete_rag_session()`

**修改示例**:
```python
# 之前
success = saver.save_text_knowledge(
    text,
    llm_service,
    entity_types=["药物", "症状", "疾病"],
    relation_types=["治疗", "导致"]
)

# 之后
success = saver.save_text_knowledge(
    text,
    llm_service
)
```

### 6. test_basic.py
**文件路径**: `test_basic.py`

**主要修改**:
- 导入 `kg_schema` 实例
- 修改 `test_llm()` 函数，移除硬编码的实体类型和关系类型
- 修改 `test_config()` 函数，增加知识图谱Schema的测试输出

### 7. demo_quick_start.py
**文件路径**: `demo_quick_start.py`

**主要修改**:
- 导入 `kg_schema` 实例
- 在快速开始演示中显示知识图谱配置信息
- 移除查询时的硬编码实体类型和关系类型

## 配置文件

### config/config.json
保持原有配置内容不变，包含：
- Neo4j连接配置
- DashScope API配置
- 嵌入服务配置
- 其他系统配置

### config/kg_schema.json
保持原有Schema内容不变，包含：
- 知识图谱名称和描述
- 实体类型定义（药物、症状、疾病等）
- 关系类型定义（治疗、导致、属于等）
- 实体关系提取提示词模板

## 优势

### 1. 配置集中管理
- 所有配置信息集中在 `config/` 目录下
- 便于统一管理和维护

### 2. 代码与配置分离
- 知识图谱Schema信息从代码中分离
- 修改实体类型或关系类型无需修改代码

### 3. 更灵活的扩展性
- 可以轻松添加新的实体类型和关系类型
- 支持不同领域的知识图谱（只需修改 `kg_schema.json`）

### 4. 更好的可维护性
- 减少了硬编码
- 提高了代码的可读性和可维护性

### 5. 向后兼容
- 所有修改都保持了向后兼容
- 仍然可以显式指定实体类型和关系类型

## 使用示例

### 基本使用（自动读取配置）
```python
from neo4j_save import Neo4jSave
from llm_service import LLMService

# 初始化服务
llm_service = LLMService()
saver = Neo4jSave(conn, embed_service)

# 自动从kg_schema.json读取实体类型和关系类型
saver.save_text_knowledge(text, llm_service)
```

### 显式指定类型（兼容旧代码）
```python
# 仍然可以显式指定实体类型和关系类型
saver.save_text_knowledge(
    text,
    llm_service,
    entity_types=["药物", "症状"],
    relation_types=["治疗"]
)
```

### 查询知识图谱Schema信息
```python
from config import kg_schema

# 获取所有实体类型
entity_types = kg_schema.get_entity_types()
print(f"实体类型: {entity_types}")

# 获取所有关系类型
relation_types = kg_schema.get_relationship_types()
print(f"关系类型: {relation_types}")

# 获取指定实体的属性
properties = kg_schema.get_entity_properties("药物")
print(f"药物属性: {properties}")
```

## 注意事项

1. **配置文件位置**: 确保 `config/config.json` 和 `config/kg_schema.json` 文件存在且格式正确

2. **实体类型一致性**: 修改 `kg_schema.json` 中的实体类型后，需要重新构建知识图谱

3. **向后兼容**: 旧代码仍然可以工作，但建议逐步迁移到新的配置方式

4. **测试**: 使用 `test_basic.py` 测试配置加载和知识图谱Schema加载

5. **扩展性**: 如需支持新领域，只需修改 `kg_schema.json` 即可
