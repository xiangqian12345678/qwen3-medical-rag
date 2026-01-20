import os
from typing import List, Dict


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
            schema_file = os.path.join(script_dir, "kg_schema.json")

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
            import json
            with open(self.schema_file, 'r', encoding='utf-8') as f:
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


kg_schema = KGSchema()
