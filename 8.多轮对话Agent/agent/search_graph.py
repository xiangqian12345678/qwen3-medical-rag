"""搜索图：用于执行单个查询的RAG检索流程"""
import logging
from functools import partial
from typing import List, Union

from langchain.output_parsers import OutputFixingParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from config.models import AppConfig
from core.utils import create_llm_client
from prompts.templates import get_prompt_template
from .tools import AgentTools
from .utils import del_think, json_to_list_document, format_document_str, _should_call_tool

logger = logging.getLogger(__name__)


# =============================================================================
# 搜索图状态定义
# =============================================================================
class SearchMessagesState(TypedDict, total=False):
    """搜索消息状态"""
    query: str  # 用户查询
    main_messages: List[Union[HumanMessage, AIMessage]]  # 主要对话消息
    other_messages: List[Union[SystemMessage, ToolMessage]]  # 其他消息
    docs: List[Document]  # 检索到的文档
    answer: str  # RAG生成的回答
    retry: int  # 剩余重试次数
    final: str  # 最终回答
    judge_result: str  # RAG质量评判结果


# =============================================================================
# 网络搜索结果模型
# =============================================================================
class NetworkSearchResult(BaseModel):
    """网络搜索结果类"""
    need_search: bool = Field(description="是否需要进行网络搜索")
    search_query: str = Field(description="网络搜索查询词", default="")
    remain_doc_index: List[int] = Field(description="保留的文档索引列表", default=[])


# =============================================================================
# 节点函数
# =============================================================================
def llm_db_search(
        state: SearchMessagesState,
        llm: BaseChatModel,
        db_tool_node: ToolNode,
        show_debug: bool
) -> SearchMessagesState:
    """数据库检索节点"""
    query = state["query"]

    if show_debug:
        logger.info(f"开始db检索节点，查询: {query}")

    db_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("call_db")["system"]),
        HumanMessage(content=get_prompt_template("call_db")["user"].format(query=query))
    ])
    state["other_messages"].append(db_ai)

    if _should_call_tool(db_ai):
        if show_debug:
            tool_calls = getattr(db_ai, 'tool_calls', None)
            if tool_calls and len(tool_calls) > 0:
                try:
                    if hasattr(tool_calls[0], 'args'):
                        args = tool_calls[0].args
                    elif isinstance(tool_calls[0], dict):
                        args = tool_calls[0].get('args', {})
                    else:
                        args = {}
                    logger.info(f"开始db检索，检索参数：{args}")
                except Exception as e:
                    logger.error(f"获取工具参数失败: {e}")

        try:
            tool_msgs: ToolMessage = db_tool_node.invoke([db_ai])
            state["other_messages"].append(tool_msgs)

            new_docs = json_to_list_document(tool_msgs[0].content)
            state["docs"].extend(new_docs)

            if show_debug:
                logger.info(f"检索到 {len(new_docs)} 条文档")
                if len(state["docs"]) >= 2:
                    logger.info(
                        f"部分示例（共{len(state['docs'])}条）：\n\n{state['docs'][0].page_content[:200]}...\n\n{state['docs'][1].page_content[:200]}..."
                    )
                elif len(state["docs"]) == 1:
                    logger.info(f"仅检索一条数据：\n\n{state['docs'][0].page_content[:200]}")
                else:
                    logger.warning("未检索到任何文档！")
        except Exception as e:
            logger.error(f"检索过程出错: {e}")

    return state


def llm_network_search(
        state: SearchMessagesState,
        judge_llm: BaseChatModel,
        network_search_llm: BaseChatModel,
        network_tool_node: ToolNode,
        show_debug: bool
) -> SearchMessagesState:
    """联网检索节点"""
    if show_debug:
        logger.info(f"检查是否缺失资料需要网络搜索...")

    parser = PydanticOutputParser(pydantic_object=NetworkSearchResult)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=judge_llm)

    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    judge_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("web_router")["system"].format(format_instructions=format_instructions)),
        ("human", get_prompt_template("web_router")["user"])
    ])

    calling_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("call_web")["system"]),
        ("human", get_prompt_template("call_web")["user"])
    ])

    judge_chain = judge_messages | judge_llm | RunnableLambda(
        lambda x: del_think(x.content)) | fixing_parser

    try:
        result: NetworkSearchResult = judge_chain.invoke({
            "query": state['query'],
            "docs": format_document_str(state.get('docs', []))
        })
        if show_debug:
            logger.info(
                f"判断结果: {'需要网络检索' if result.need_search else '不需要网络检索'}, 检索文本：{result.search_query}"
            )

        judge_ai_content = f"分析结果: {result.model_dump()}"
        judge_ai = AIMessage(content=judge_ai_content)
        state["other_messages"].append(judge_ai)

    except Exception as e:
        logger.error(f"JSON解析错误: {e}")
        result = NetworkSearchResult(need_search=False, search_query="", remain_doc_index=[])
        judge_ai = AIMessage(content=f"解析失败，使用默认值: {result.model_dump()}")
        state["other_messages"].append(judge_ai)

    if result.need_search and result.search_query.strip():
        search_chain = calling_messages | network_search_llm
        search_ai = search_chain.invoke({"search_query": result.search_query})
        state["other_messages"].append(search_ai)

        if _should_call_tool(search_ai):
            tool_msgs: ToolMessage = network_tool_node.invoke([search_ai])
            state["other_messages"].append(tool_msgs)

            remain_doc = result.remain_doc_index
            if remain_doc:
                valid_indices = [i - 1 for i in remain_doc if 0 < i <= len(state.get("docs", []))]
                state["docs"] = [state["docs"][i] for i in valid_indices]
            else:
                state["docs"] = []

            state["docs"].extend(json_to_list_document(tool_msgs[0].content))
            if show_debug:
                logger.info(f"网络检索完毕")
    else:
        if show_debug:
            logger.info(f"信息完整，无需网络搜索...")

    return state


def rag_node(
        state: SearchMessagesState,
        llm: BaseChatModel,
        show_debug: bool
) -> SearchMessagesState:
    """RAG生成节点"""
    if show_debug:
        logger.info(f"开始RAG...当前文档数量: {len(state.get('docs', []))}")

    sys = get_prompt_template("basic_rag")["system"]
    user = get_prompt_template("basic_rag")["user"]

    prompt = [
        SystemMessage(content=sys),
        HumanMessage(
            content=user.format(
                all_document_str=format_document_str(state.get("docs", [])),
                input=state["query"]
            )
        )
    ]

    rag_response = llm.invoke(prompt)

    # 调试：记录 LLM 原始输出
    if show_debug:
        logger.info(f"LLM原始输出长度: {len(rag_response.content)}")
        logger.info(f"LLM原始输出(前200字符): {rag_response.content[:200]}")

    rag_content = del_think(rag_response.content)

    # 调试：记录 del_think 后的输出
    if show_debug:
        logger.info(f"del_think后长度: {len(rag_content)}")
        logger.info(f"del_think后内容(前200字符): {rag_content[:200]}")

    rag_ai = AIMessage(content=rag_content)

    if show_debug:
        logger.info(f"RAG回答: {rag_ai.content}")

    if not isinstance(state["main_messages"][-1], AIMessage):
        state["main_messages"].append(rag_ai)
    else:
        state["main_messages"].pop()
        state["main_messages"].append(rag_ai)

    state["answer"] = rag_ai.content
    return state


def judge_node(
        state: SearchMessagesState,
        llm: BaseChatModel,
        show_debug: bool
) -> SearchMessagesState:
    """质量判断节点"""
    if show_debug:
        logger.info(f"开始评估...")

    judge_ai = llm.invoke([
        SystemMessage(content=get_prompt_template("judge_rag")["system"]),
        HumanMessage(
            content=get_prompt_template("judge_rag")["user"].format(
                format_document_str=format_document_str(state.get('docs', [])),
                query=state['query'],
                summary=state.get('answer', '')
            )
        )
    ])

    result = del_think(judge_ai.content or "").strip().lower()

    if show_debug:
        logger.info(f"评估结果: {result[:20]}")

    state["other_messages"].append(AIMessage(content=f"[JUDGE]={result}"))

    if 'y' in result:
        state["judge_result"] = "pass"
    else:
        retries_left = int(state.get("retry", 0))
        if retries_left > 0:
            state["retry"] = retries_left - 1
            state["judge_result"] = "retry"
        else:
            state["judge_result"] = "fail"

    return state


def finish_success(state: SearchMessagesState) -> SearchMessagesState:
    """成功结束节点"""
    state["final"] = (state.get("answer", "") or "").strip() or "（空）"
    return state


def finish_fail(state: SearchMessagesState) -> SearchMessagesState:
    """失败结束节点"""
    base = (state.get("answer", "") or "").strip() or "（空）"
    state["final"] = base + "\n\n（内容可能不属实）"
    return state


# =============================================================================
# 搜索图类
# =============================================================================
class SearchGraph:
    """搜索图：执行单个查询的RAG检索流程"""

    def __init__(self, config: AppConfig, power_model: BaseChatModel, websearch_func=None) -> None:
        """
        初始化搜索图

        Args:
            config: 应用配置
            power_model: 强大模型实例（用于工具调用）
            websearch_func: 网络搜索函数（可选）
        """
        self.config = config

        self.agent_tools = AgentTools(config)
        if websearch_func and config.agent.network_search_enabled:
            self.agent_tools.register_websearch(websearch_func)

        self.db_search_tool = self.agent_tools.make_database_search_tool()

        # 只有在启用网络搜索时才创建网络搜索工具
        if config.agent.network_search_enabled:
            self.network_search_tool = self.agent_tools.make_web_search_tool()
            self.network_search_llm = power_model.bind_tools([self.network_search_tool])
            self.network_tool_node = ToolNode([self.network_search_tool])
        else:
            self.network_search_tool = None
            self.network_search_llm = None
            self.network_tool_node = None

        self.db_search_llm = power_model.bind_tools([self.db_search_tool])

        self.llm = create_llm_client(config.llm)

        self.db_tool_node = ToolNode([self.db_search_tool])

        self.search_graph = None

    def build_search_graph(self):
        """构建搜索图"""

        def judge_router(state: SearchMessagesState) -> str:
            """路由函数"""
            return state.get("judge_result", "fail")

        g = StateGraph(SearchMessagesState)

        db_search_node_func = partial(
            llm_db_search,
            llm=self.db_search_llm,
            db_tool_node=self.db_tool_node,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("db_search", db_search_node_func)

        # 只在启用网络搜索时添加web_search节点
        if self.config.agent.network_search_enabled:
            network_search_node_func = partial(
                llm_network_search,
                judge_llm=self.llm,
                network_search_llm=self.network_search_llm,
                network_tool_node=self.network_tool_node,
                show_debug=self.config.multi_dialogue_rag.console_debug
            )
            g.add_node("web_search", network_search_node_func)

        rag_node_func = partial(
            rag_node,
            llm=self.llm,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("rag", rag_node_func)

        g.add_node("finish_success", finish_success)
        g.add_node("finish_fail", finish_fail)

        if self.config.agent.mode == "analysis":
            judge_node_func = partial(
                judge_node,
                llm=self.llm,
                show_debug=self.config.multi_dialogue_rag.console_debug
            )
            g.add_node("judge", judge_node_func)

            if self.config.agent.network_search_enabled:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "web_search")
                g.add_edge("web_search", "rag")
            else:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "rag")

            g.add_edge("rag", "judge")
            g.add_conditional_edges(
                "judge",
                judge_router,
                {
                    "pass": "finish_success",
                    "retry": "rag",
                    "fail": "finish_fail"
                }
            )
            g.add_edge("finish_success", END)
            g.add_edge("finish_fail", END)

        elif self.config.agent.mode == "fast":
            if self.config.agent.network_search_enabled:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "web_search")
                g.add_edge("web_search", "rag")
            else:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "rag")
            g.add_edge("rag", END)

        else:  # normal模式
            if self.config.agent.network_search_enabled:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "web_search")
                g.add_edge("web_search", "rag")
            else:
                g.set_entry_point("db_search")
                g.add_edge("db_search", "rag")
            g.add_edge("rag", END)

        self.search_graph = g.compile()

    def run(self, init_state: SearchMessagesState) -> SearchMessagesState:
        """
        执行搜索图

        Args:
            init_state: 初始状态

        Returns:
            执行完成后的最终状态
        """
        if self.search_graph is None:
            self.build_search_graph()

        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state

    def answer(self, query: str) -> str:
        """
        回答用户问题

        Args:
            query: 用户问题

        Returns:
            回答结果
        """
        if self.search_graph is None:
            self.build_search_graph()

        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "answer": "",
            "retry": self.config.agent.max_attempts,
            "final": "",
            "judge_result": ""
        }

        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state.get("final", "") or out_state.get("answer", "") or "（空）"
