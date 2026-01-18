"""
配置管理模块
"""
import os
from typing import List, Dict, Any

import yaml


class Config:
    """配置类，负责加载和管理配置参数"""

    def __init__(self, config_file: str = None):
        """
        初始化配置

        【输入示例】
        config = Config("config/kg_config.yaml")

        【输出示例】
        None (配置对象已初始化)
        """
        # 如果没有指定配置文件，使用相对于脚本位置的路径
        if config_file is None:
            # 获取脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(script_dir, "config", "kg_config.yaml")

        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        从YAML文件加载配置

        【输入示例】
        无

        【输出示例】
        {
            "neo4j": {...},
            "llm": {...},
            "embedding": {...},
            ...
        }
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def get(self, key: str, default=None):
        """
        获取配置项

        【输入示例】
        config.get("neo4j.uri", "bolt://localhost:7687")

        【输出示例】
        "bolt://localhost:7687"
        """
        # 支持点号分隔的路径，如 "neo4j.uri"
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        return self.config.get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置区块

        【输入示例】
        neo4j_config = config.get_section("neo4j")

        【输出示例】
        {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "12345678",
            ...
        }
        """
        return self.config.get(section, {})

    def __getattr__(self, name: str):
        """通过属性方式访问配置"""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class Neo4jConfig:
    """Neo4j数据库配置类"""

    def __init__(self, config: Config = None):
        """
        初始化Neo4j配置

        【输入示例】
        neo4j_config = Neo4jConfig(config)
        """
        self.config = config or Config()
        self.neo4j_config = self.config.get_section("neo4j")

    @property
    def uri(self) -> str:
        return self.neo4j_config.get("uri", "bolt://localhost:7687")

    @property
    def user(self) -> str:
        return self.neo4j_config.get("user", "neo4j")

    @property
    def password(self) -> str:
        return self.neo4j_config.get("password", "")

    @property
    def database(self) -> str:
        return self.neo4j_config.get("database", "neo4j")

    @property
    def max_connection_lifetime(self) -> int:
        return self.neo4j_config.get("max_connection_lifetime", 3600)

    @property
    def max_connection_pool_size(self) -> int:
        return self.neo4j_config.get("max_connection_pool_size", 50)

    @property
    def connection_timeout(self) -> float:
        return self.neo4j_config.get("connection_timeout", 30.0)

    @property
    def acquisition_timeout(self) -> float:
        return self.neo4j_config.get("acquisition_timeout", 60.0)


class LLMConfig:
    """大模型配置类"""

    def __init__(self, config: Config = None):
        """
        初始化大模型配置

        【输入示例】
        llm_config = LLMConfig(config)
        """
        self.config = config or Config()
        self.llm_config = self.config.get_section("llm")

    @property
    def provider(self) -> str:
        return self.llm_config.get("provider", "dashscope")

    @property
    def api_key(self) -> str:
        return self.llm_config.get("api_key", "")

    @property
    def base_url(self) -> str:
        return self.llm_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    @property
    def model(self) -> str:
        return self.llm_config.get("model", "qwen-plus")

    @property
    def temperature(self) -> float:
        return self.llm_config.get("temperature", 0.3)

    @property
    def max_tokens(self) -> int:
        return self.llm_config.get("max_tokens", 2048)


class EmbeddingConfig:
    """嵌入模型配置类"""

    def __init__(self, config: Config = None):
        """
        初始化嵌入配置

        【输入示例】
        embedding_config = EmbeddingConfig(config)
        """
        self.config = config or Config()
        self.embedding_config = self.config.get_section("embedding")

    @property
    def provider(self) -> str:
        return self.embedding_config.get("provider", "dashscope")

    @property
    def base_url(self) -> str:
        return self.embedding_config.get("base_url", "")

    @property
    def model(self) -> str:
        return self.embedding_config.get("model", "text-embedding-v2")

    @property
    def dimension(self) -> int:
        return self.embedding_config.get("dimension", 1536)


class KGConfig:
    """知识图谱配置类"""

    def __init__(self, config: Config = None):
        """
        初始化知识图谱配置

        【输入示例】
        kg_config = KGConfig(config)
        """
        self.config = config or Config()
        self.kg_config = self.config.get_section("knowledge_graph")

    @property
    def schema_file(self) -> str:
        return self.kg_config.get("schema_file", "config/kg_schema.json")

    @property
    def vector_index_path(self) -> str:
        return self.kg_config.get("vector_index_path", "knowledge_index/kg_vector_index.pkl")

    @property
    def chunk_size(self) -> int:
        return self.kg_config.get("chunk_size", 1000)

    @property
    def chunk_overlap(self) -> int:
        return self.kg_config.get("chunk_overlap", 100)

    @property
    def graph_top_k(self) -> int:
        return self.kg_config.get("graph_top_k", 10)

    @property
    def similarity_threshold(self) -> float:
        return self.kg_config.get("similarity_threshold", 0.7)


class RAGConfig:
    """RAG系统配置类"""

    def __init__(self, config: Config = None):
        """
        初始化RAG配置

        【输入示例】
        rag_config = RAGConfig(config)
        """
        self.config = config or Config()
        self.rag_config = self.config.get_section("rag")

    @property
    def depth(self) -> int:
        return self.rag_config.get("depth", 2)

    @property
    def top_k(self) -> int:
        return self.rag_config.get("top_k", 5)

    @property
    def max_tokens(self) -> int:
        return self.rag_config.get("max_tokens", 1024)

    @property
    def cache_enabled(self) -> bool:
        return self.rag_config.get("cache_enabled", True)

    @property
    def console_debug(self) -> bool:
        return self.rag_config.get("console_debug", True)


class KGSchema:
    """知识图谱Schema配置类"""

    def __init__(self, schema_file: str = None):
        """
        初始化知识图谱Schema

        【输入示例】
        kg_schema = KGSchema("config/kg_schema.json")

        【输出示例】
        None (Schema对象已初始化)
        """
        if schema_file is None:
            # 获取脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            schema_file = os.path.join(script_dir, "config", "kg_schema.json")

        self.schema_file = schema_file
        self.schema = self._load_schema()

    def _load_schema(self) -> dict:
        """
        从JSON文件加载知识图谱Schema

        【输入示例】
        无

        【输出示例】
        {
            "name": "医疗知识图谱",
            "entity_types": [...],
            "relationship_types": [...]
        }
        """
        if os.path.exists(self.schema_file):
            with open(self.schema_file, 'r', encoding='utf-8') as f:
                import json
                return json.load(f)
        return {}

    def get_entity_types(self) -> List[str]:
        """
        获取所有实体类型

        【输出示例】
        ["药物", "症状", "疾病"]
        """
        entity_types = self.schema.get("entity_types", [])
        return [et["name"] for et in entity_types]

    def get_entity_properties(self, entity_type: str) -> List[str]:
        """
        获取指定实体类型的属性列表

        【输入示例】
        properties = kg_schema.get_entity_properties("药物")

        【输出示例】
        ["名称", "成分", "剂量", "适应症", "副作用"]
        """
        entity_types = self.schema.get("entity_types", [])
        for et in entity_types:
            if et["name"] == entity_type:
                return et.get("properties", [])
        return []

    def get_relationship_types(self) -> List[str]:
        """
        获取所有关系类型

        【输出示例】
        ["治疗", "导致", "属于"]
        """
        relationship_types = self.schema.get("relationship_types", [])
        return [rt["name"] for rt in relationship_types]

    def get_relationship_info(self, rel_type: str) -> Dict:
        """
        获取指定关系类型的详细信息

        【输入示例】
        info = kg_schema.get_relationship_info("治疗")

        【输出示例】
        {
            "name": "治疗",
            "source": ["药物"],
            "target": ["症状", "疾病"]
        }
        """
        relationship_types = self.schema.get("relationship_types", [])
        for rt in relationship_types:
            if rt["name"] == rel_type:
                return rt
        return {}

    def get_extraction_prompt(self) -> str:
        """
        获取实体关系提取提示词

        【输出示例】
        "你是一个专业的知识图谱工程师..."
        """
        return self.schema.get("extraction_prompt", "")

    def format_extraction_prompt(self, text: str) -> str:
        """
        格式化提取提示词，替换占位符

        【输入示例】
        prompt = kg_schema.format_extraction_prompt("阿司匹林治疗头痛")

        【输出示例】
        "你是一个专业的知识图谱工程师... 实体类型必须是：药物、症状、疾病..."
        """
        template = self.get_extraction_prompt()
        entity_types = "、".join(self.get_entity_types())
        relationship_types = "、".join(self.get_relationship_types())

        return template.format(
            entity_types=entity_types,
            relationship_types=relationship_types,
            text=text
        )

    def get_all_entity_info(self) -> List[Dict]:
        """
        获取所有实体类型信息

        【输出示例】
        [
            {"name": "药物", "properties": ["名称", "成分", "剂量"]},
            {"name": "症状", "properties": ["名称", "描述", "严重程度"]}
        ]
        """
        return self.schema.get("entity_types", [])

    def get_all_relationship_info(self) -> List[Dict]:
        """
        获取所有关系类型信息

        【输出示例】
        [
            {"name": "治疗", "source": ["药物"], "target": ["症状", "疾病"]},
            {"name": "导致", "source": ["疾病"], "target": ["症状"]}
        ]
        """
        return self.schema.get("relationship_types", [])


# 创建全局配置实例
config = Config()
neo4j_config = Neo4jConfig(config)
llm_config = LLMConfig(config)
embedding_config = EmbeddingConfig(config)
kg_config = KGConfig(config)
rag_config = RAGConfig(config)
kg_schema = KGSchema()
