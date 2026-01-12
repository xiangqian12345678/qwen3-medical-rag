import ast
import json
import logging
import operator as op
from typing import List

from langchain.tools import tool
from langchain_core.documents import Document

from ...config.models import AppConfig
from ...config.models import SearchRequest
from ...core.DBFactory import get_kb

logger = logging.getLogger(__name__)


class AgentTools:
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        self.WEBSEARCH_FUNC = None

    def register_websearch(self, func):
        self.WEBSEARCH_FUNC = func

    def make_web_search_tool(self):
        if self.WEBSEARCH_FUNC is None:
            raise "未注册网络检索工具"

        cnt = self.app_config.agent.network_search_cnt

        @tool("web_search")
        def web_search(query: str) -> str:
            """
            联网搜索工具（LangChain Tool）
            Tool 名称：
                web_search
                （该名称会暴露给 LLM，用于 tool_calls）
            输入参数：
                query (str)
                    - 由 LLM 生成或优化后的搜索查询词
                    - 通常比用户原始 query 更适合搜索引擎
                    - 例如："2024 高血压 治疗 指南 最新"
            输出：
                str（JSON 字符串）
                    - JSON 数组格式
                    - 每个元素是一个 Document 的字典表示
                    - 结构示例：
                      [
                        {
                          "page_content": "...",
                          "metadata": {
                            "source": "...",
                            "score": 0.92
                          }
                        },
                        ...
                      ]

            调用链位置：
                LLM(tool_call) → ToolNode → web_search → ToolMessage → json_to_list_document
            """

            # ---------- 1. 调用真实的联网搜索函数 ----------
            # self.WEBSEARCH_FUNC:
            #   - 通常是腾讯云 / Bing / Google / 自建爬虫等
            #   - 返回 List[Document]
            # 参数说明：
            #   query: 搜索关键词
            #   cnt:   返回文档数量（来自外部配置或上下文）
            results: List[Document] = self.WEBSEARCH_FUNC(query, cnt)

            # ---------- 2. 将 Document 对象序列化为 JSON ----------
            # d.model_dump():
            #   - Pydantic / LangChain Document 转 dict
            #   - 包含 page_content / metadata 等字段
            # ensure_ascii=False:
            #   - 保证中文不被转义为 \uXXXX
            #   - 方便后续日志和调试
            return json.dumps(
                [d.model_dump() for d in results],
                ensure_ascii=False
            )

        return web_search

    def make_database_search_tool(self):
        """
        构造一个「数据库检索 Tool」，用于注入到 LLM / Agent / Function Calling 环境中

        设计目的：
        1. 将本地向量数据库 / 混合检索能力，封装为一个标准 Tool
        2. 供 Agent 在推理过程中按需调用（如 LangChain / OpenAI tools）
        3. 隐藏底层 KB 获取、配置管理与检索实现细节

        返回：
        - Callable:
            一个已经通过 @tool 装饰的函数对象，可直接注册到 Agent
        """

        @tool("database_search")
        def database_search(search_config: SearchRequest) -> str:
            """
            使用本地向量数据库进行检索的 Tool 实现
            Tool 名称：
            - "database_search"
              该名称会暴露给 LLM，用于在推理过程中选择是否调用该工具
            输入参数：
            - search_config (SearchRequest):
                检索请求对象，通常由 LLM 按 Schema 自动构造
                常见字段可能包括：
                - query: str                  # 用户查询文本
                - top_k: int                  # 返回结果数量
                - filters / metadata: dict    # 结构化过滤条件
                - search_type: enum           # dense / sparse / hybrid 等
            输出：
            - str (JSON):
                检索结果的 JSON 字符串表示，格式为：
                [
                    {
                        ... Document / SearchResult 的字段 ...
                    },
                    ...
                ]

                之所以返回 str 而非 Python 对象，是因为：
                - Tool 调用结果最终会回传给 LLM
                - LLM 更擅长消费 JSON 文本而非 Python 原生对象
            """

            # 1. 从当前应用配置中获取 KnowledgeBase 实例
            #    get_kb 内部使用 (pid + config) 做进程级单例缓存：
            #    - 同一进程内多次调用不会重复初始化 KB
            #    - 多进程部署时各进程互不干扰
            kb = get_kb(self.app_config.model_dump())

            # 2. 调用 KnowledgeBase 的 search 方法执行实际检索
            #    - req 为标准化的 SearchRequest
            #    - 返回值通常为 List[SearchResult / Document]
            results = kb.search(req=search_config)

            # 3. 将检索结果序列化为 JSON 字符串
            #    - 对每个结果调用 model_dump()（Pydantic 模型）
            #    - ensure_ascii=False 保证中文不被转义
            #    - 返回给 LLM 作为 Tool 执行结果
            return json.dumps(
                [d.model_dump() for d in results],
                ensure_ascii=False
            )

        # 4. 返回 Tool 函数本身，供外部注册 / 注入 Agent
        return database_search

