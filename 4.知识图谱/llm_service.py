"""
大模型服务模块
负责调用大模型API进行实体关系提取和问答生成
"""
from typing import Dict, List
from openai import OpenAI
import httpx
import re
import json
from config import config, kg_schema


class LLMService:
    """
    大模型服务类
    封装对通义千问等大模型的调用
    """

    def __init__(self, api_key: str = None):
        """
        初始化大模型服务

        【输入示例】
        service = LLMService(api_key="sk-xxx")

        【输出示例】
        None (服务已初始化)
        """
        self.api_key = api_key or config.DASHSCOPE_API_KEY

        # 创建OpenAI兼容客户端
        http_client = httpx.Client(
            trust_env=False,
            timeout=60.0
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=config.DASHSCOPE_API_BASE,
            http_client=http_client
        )

        self.model = "qwen-plus"

    def extract_entities_relations(self, text: str, entity_types: List[str] = None,
                                   relation_types: List[str] = None) -> Dict:
        """
        从文本中提取实体和关系

        【输入示例】
        text = "阿司匹林可以治疗头痛和发热"
        entity_types = ["药物", "症状", "疾病"]
        relation_types = ["治疗", "导致"]
        result = service.extract_entities_relations(text, entity_types, relation_types)
        或
        result = service.extract_entities_relations(text)  # 自动从kg_schema.json读取

        【输出示例】
        {
            "entities": [
                {"id": "e1", "name": "阿司匹林", "type": "药物", "properties": {}},
                {"id": "e2", "name": "头痛", "type": "症状", "properties": {}},
                {"id": "e3", "name": "发热", "type": "症状", "properties": {}}
            ],
            "relationships": [
                {"source": "e1", "target": "e2", "type": "治疗", "properties": {}},
                {"source": "e1", "target": "e3", "type": "治疗", "properties": {}}
            ]
        }
        """
        # 如果没有指定实体类型和关系类型，从kg_schema.json读取
        if entity_types is None:
            entity_types = kg_schema.get_entity_types()
        if relation_types is None:
            relation_types = kg_schema.get_relationship_types()

        # 构建提示词
        prompt = f"""
你是一个专业的知识图谱工程师，请从以下文本中提取实体和关系，严格按照以下要求：
1. 只提取文本中明确提到的实体和关系
2. 使用JSON格式输出，包含两个列表："entities"和"relationships"
3. 实体格式：{{"id": "唯一ID", "name": "实体名称", "type": "实体类型", "properties": {{"属性名": "属性值"}}}}
4. 关系格式：{{"source": "源实体ID", "target": "目标实体ID", "type": "关系类型", "properties": {{"属性名": "属性值"}}}}
5. 实体类型必须是以下之一：{', '.join(entity_types)}
6. 关系类型必须是以下之一：{', '.join(relation_types)}

文本内容：
{text}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识图谱工程师"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "text"}
            )

            content = response.choices[0].message.content

            # 解析JSON
            return self._parse_json_response(content)

        except Exception as e:
            print(f"提取实体关系失败: {e}")
            return {"entities": [], "relationships": []}

    def _parse_json_response(self, content: str) -> Dict:
        """
        解析JSON格式的响应

        【输入示例】
        content = '{"entities": [...], "relationships": [...]}'

        【输出示例】
        {"entities": [...], "relationships": [...]}
        """
        try:
            # 查找JSON部分
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                print(f"未找到JSON内容: {content[:200]}...")
                return {"entities": [], "relationships": []}

            json_str = content[json_start:json_end]

            # 修复常见的JSON格式问题
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r"(\w+):", r'"\1":', json_str)

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始内容: {content[:200]}...")
            return {"entities": [], "relationships": []}

    def generate_answer(self, question: str, context: str = "",
                       max_tokens: int = 1024) -> str:
        """
        生成问答答案

        【输入示例】
        question = "阿司匹林有什么作用？"
        context = "阿司匹林是一种非甾体抗炎药，具有镇痛、解热、抗炎作用..."
        answer = service.generate_answer(question, context)

        【输出示例】
        "阿司匹林主要用于镇痛、解热和抗炎作用，常用于治疗头痛、关节痛、发热等症状..."
        """
        prompt = f"""
请基于以下背景信息回答问题，如果背景信息不足以回答问题，请说明：

背景信息：
{context}

问题：
{question}

请给出专业、准确的回答：
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识问答助手"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"生成答案失败: {e}")
            return "抱歉，生成答案时出现问题，请稍后再试。"

    def generate_rag_answer(self, question: str, kg_results: List[Dict],
                            vdb_results: List[str] = None) -> str:
        """
        生成RAG整合答案

        【输入示例】
        question = "阿司匹林可以治疗什么？"
        kg_results = [{"source": "阿司匹林", "target": "头痛", "type": "治疗"}]
        vdb_results = ["阿司匹林是一种常用药物..."]
        answer = service.generate_rag_answer(question, kg_results, vdb_results)

        【输出示例】
        "根据知识图谱，阿司匹林与头痛之间存在治疗关系..."
        """
        # 构建知识图谱描述
        kg_desc = ""
        if kg_results:
            kg_desc = "知识图谱信息：\n"
            for record in kg_results[:5]:
                kg_desc += f"- {record.get('source', '')} {record.get('relationship', '')} {record.get('target', '')}\n"

        # 构建向量检索描述
        vdb_desc = ""
        if vdb_results:
            vdb_desc = "\n相关文档片段：\n"
            for i, doc in enumerate(vdb_results[:3]):
                vdb_desc += f"片段{i+1}: {doc[:200]}...\n"

        prompt = f"""
你是一个专业的知识问答助手，请基于以下信息回答用户问题：

{kg_desc}

{vdb_desc}

用户问题：
{question}

请给出专业、准确的回答：
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识问答助手"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.35
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"生成RAG答案失败: {e}")
            return "抱歉，生成答案时出现问题，请稍后再试。"

    def close(self):
        """
        关闭客户端

        【输入示例】
        service.close()

        【输出示例】
        None (客户端已关闭)
        """
        self.client.close()


# 使用示例
if __name__ == "__main__":
    # 示例1: 提取实体关系
    print("示例1: 提取实体关系")
    service = LLMService()
    text = "阿司匹林是一种常用药物，主要用于治疗头痛和发热。"
    entity_types = ["药物", "症状", "疾病"]
    relation_types = ["治疗", "导致", "属于"]

    result = service.extract_entities_relations(text, entity_types, relation_types)
    print(f"提取到 {len(result['entities'])} 个实体")
    print(f"提取到 {len(result['relationships'])} 个关系")
    for entity in result['entities']:
        print(f"  - {entity['type']}: {entity['name']}")

    # 示例2: 生成答案
    print("\n示例2: 生成答案")
    question = "阿司匹林有什么作用？"
    context = "阿司匹林是一种非甾体抗炎药，具有镇痛、解热、抗炎作用，常用于治疗头痛、关节痛、发热等症状。"
    answer = service.generate_answer(question, context)
    print(f"问题: {question}")
    print(f"答案: {answer}")

    service.close()
