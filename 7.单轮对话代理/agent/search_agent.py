"""单轮对话Agent"""
import json
import logging
import re
from functools import partial
from typing import List, Union

from config.models import AppConfig
from core.utils import create_llm_client, format_documents
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from prompts.templates import get_prompt_template
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .tools import AgentTools

logger = logging.getLogger(__name__)


# =============================================================================
# 工具函数
# =============================================================================
def del_think(text: str) -> str:
    """移除思考过程标签"""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def json_to_list_document(text: str) -> List[Document]:
    """将JSON转换为Document列表"""
    return [Document(**d) for d in json.loads(text)]


def format_document_str(documents: List[Document]) -> str:
    """格式化文档为字符串"""
    parts = []
    for i, d in enumerate(reversed(documents)):
        if i >= 6:
            break
        parts.append(f"## 文档{i + 1}：\n{d.page_content}\n")
    return "".join(parts)


# =============================================================================
# 状态定义
# =============================================================================
class SearchMessagesState(TypedDict, total=False):
    """搜索消息状态"""
    query: str
    main_messages: List[Union[HumanMessage, AIMessage]]
    other_messages: List[Union[SystemMessage, ToolMessage]]
    docs: List[Document]
    summary: str
    retry: int
    final: str
    judge_result: str


# =============================================================================
# 判断结果模型
# =============================================================================
class JudgeResult(BaseModel):
    """判断结果模型"""
    pass_fail: bool = Field(description="是否通过")


# =============================================================================
# 节点函数
# =============================================================================
def _should_call_tool(last_ai) -> bool:
    """判断是否需要调用工具"""
    return bool(getattr(last_ai, "tool_calls", None))


def db_search_node(
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

    # 获取 tool_calls - 兼容不同的格式
    tool_calls = getattr(db_ai, 'tool_calls', None)
    if show_debug:
        logger.info(f"LLM调用返回，tool_calls: {tool_calls}")
        logger.info(f"LLM消息类型: {type(db_ai)}")
        logger.info(f"LLM消息属性: {dir(db_ai)}")

    if _should_call_tool(db_ai):
        if show_debug:
            # 获取工具调用参数
            if tool_calls and len(tool_calls) > 0:
                # 兼容 LangChain ToolCall 对象和字典格式
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

            # 解析检索结果
            if show_debug:
                logger.info(f"工具返回消息类型: {type(tool_msgs)}")
                logger.info(f"工具返回内容: {tool_msgs[0].content}")

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
            import traceback
            traceback.print_exc()
    else:
        if show_debug:
            logger.warning("LLM未调用检索工具！")

    return state


def rag_node(
        state: SearchMessagesState,
        llm: BaseChatModel,
        show_debug: bool
) -> SearchMessagesState:
    """RAG生成节点"""
    if show_debug:
        logger.info(f"开始RAG...当前文档数量: {len(state.get('docs', []))}")
        doc_str = format_document_str(state.get("docs", []))
        logger.info(f"文档内容预览:\n{doc_str[:500]}..." if len(doc_str) > 500 else f"文档内容:\n{doc_str}")

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

    rag_ai = llm.invoke(prompt)
    raw_content = rag_ai.content
    rag_ai.content = del_think(rag_ai.content)

    if show_debug:
        logger.info(f"RAG原始回答: {raw_content}")
        logger.info(f"RAG处理后回答: {rag_ai.content}")

    if not isinstance(state["main_messages"][-1], AIMessage):
        state["main_messages"].append(rag_ai)
    else:
        state["main_messages"].pop()
        state["main_messages"].append(rag_ai)

    state["summary"] = rag_ai.content
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
                summary=state.get('summary', '')
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
    state["final"] = (state.get("summary", "") or "").strip() or "（空）"
    return state


def finish_fail(state: SearchMessagesState) -> SearchMessagesState:
    """失败结束节点"""
    base = (state.get("summary", "") or "").strip() or "（空）"
    state["final"] = base + "\n\n（内容可能不属实）"
    return state


# =============================================================================
# 单轮对话Agent类
# =============================================================================
class SingleDialogueAgent:
    """单轮对话Agent"""

    def __init__(self, config: AppConfig, power_model: BaseChatModel, websearch_func=None):
        """
        初始化Agent

        Args:
            config: 应用配置
            power_model: 强大模型实例（用于工具调用）
            websearch_func: 网络搜索函数（可选）
        """
        self.config = config
        self.power_model = power_model
        self.llm = create_llm_client(self.config.llm)

        # 初始化工具
        self.agent_tools = AgentTools(self.config)
        if websearch_func:
            self.agent_tools.register_websearch(websearch_func)

        # 创建工具
        self.db_search_tool = self.agent_tools.make_database_search_tool()
        self.db_search_llm = power_model.bind_tools([self.db_search_tool])
        self.db_tool_node = ToolNode([self.db_search_tool])

        # 构建图
        self._build_graph()

    def _build_graph(self):
        """构建搜索图"""
        def judge_router(state: SearchMessagesState) -> str:
            return state.get("judge_result", "fail")

        g = StateGraph(SearchMessagesState)

        # 添加节点
        db_search_node_func = partial(
            db_search_node,
            llm=self.db_search_llm,
            db_tool_node=self.db_tool_node,
            show_debug=self.config.agent.console_debug
        )
        g.add_node("db_search", db_search_node_func)

        rag_node_func = partial(
            rag_node,
            llm=self.llm,
            show_debug=self.config.agent.console_debug
        )
        g.add_node("rag", rag_node_func)

        g.add_node("finish_success", finish_success)
        g.add_node("finish_fail", finish_fail)

        # 根据模式添加节点
        if self.config.agent.mode == "analysis":
            judge_node_func = partial(
                judge_node,
                llm=self.llm,
                show_debug=self.config.agent.console_debug
            )
            g.add_node("judge", judge_node_func)

            # 设置边
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
            g.set_entry_point("db_search")
            g.add_edge("db_search", "rag")
            g.add_edge("rag", END)

        else:  # normal模式
            g.set_entry_point("db_search")
            g.add_edge("db_search", "rag")
            g.add_edge("rag", END)

        self.search_graph = g.compile()

    def answer(self, query: str) -> str:
        """
        回答用户问题

        Args:
            query: 用户问题

        Returns:
            回答结果
        """
        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "summary": "",
            "retry": self.config.agent.max_attempts,
            "final": "",
            "judge_result": ""
        }

        out_state: SearchMessagesState = self.search_graph.invoke(init_state)
        return out_state.get("final", "") or out_state.get("summary", "") or "（空）"
