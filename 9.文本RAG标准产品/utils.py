"""工具函数"""
import httpx
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

project_dir = Path(__file__).parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from rag_config import LLMConfig

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    pass


def create_llm_client(config: LLMConfig) -> BaseChatModel:
    """创建LLM客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        return ChatOpenAI(**kwargs)

    elif config.provider == "ollama":
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.max_tokens:
            kwargs["num_predict"] = config.max_tokens
        logger.info(f"创建 Ollama LLM: {config.model}, base_url: {config.base_url}")
        return ChatOllama(**kwargs)

    elif config.provider == "dashscope":
        # 创建 HTTP 客户端，参考知识图谱模块的配置
        http_client = httpx.Client(
            trust_env=False,
            timeout=60.0
        )

        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
            "http_client": http_client
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        logger.info(f"创建 DashScope LLM: {config.model}")
        return ChatTongyi(**kwargs)

    else:
        raise ValueError(f"不支持的LLM提供商: {config.provider}")




def format_documents(documents) -> str:
    """格式化文档为字符串"""
    parts = []
    for i, doc in enumerate(documents):
        if hasattr(doc, 'page_content'):
            content = doc.page_content
        elif isinstance(doc, dict):
            content = doc.get('page_content', str(doc))
        else:
            content = str(doc)
        parts.append(f"## 文档{i + 1}：\n{content}\n")
    return "".join(parts)


def strip_think_get_tokens(msg: AIMessage):
    text = msg.content
    msg_len = len(msg.content)

    try:
        msg_token_len = msg.usage_metadata["output_tokens"]
    except Exception as e1:
        try:
            msg_token_len = msg.response_metadata["token_usage"]["output_tokens"]
        except Exception as e2:
            msg_token_len = 0

    dur = msg.response_metadata.get("total_duration", 0) / 1e9

    return {
        "msg": re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip(),
        "msg_len": msg_len,
        "msg_token_len": msg_token_len,
        "generate_time": dur
    }


def del_think(text: str) -> str:
    """移除思考过程标签"""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def json_to_list_document(text: str):
    """将JSON转换为Document列表"""
    import json
    from langchain_core.documents import Document

    # 检查输入是否为空
    if not text or not text.strip():
        logger.warning("json_to_list_document: 输入为空字符串")
        return []

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            logger.warning(f"json_to_list_document: 期望列表,得到 {type(data)}")
            return []
        return [Document(**d) for d in data]
    except json.JSONDecodeError as e:
        logger.error(f"json_to_list_document: JSON解析失败 - {e}")
        logger.error(f"输入内容: {text[:500] if len(text) > 500 else text}")
        return []
    except Exception as e:
        logger.error(f"json_to_list_document: 转换失败 - {e}")
        return []


def format_document_str(documents) -> str:
    """格式化文档为字符串"""
    parts = []
    for i, d in enumerate(reversed(documents)):
        if i >= 6:
            break
        parts.append(f"## 文档{i + 1}：\n{d.page_content}\n")
    return "".join(parts)


def _should_call_tool(last_ai) -> bool:
    """判断是否需要调用工具"""
    return bool(getattr(last_ai, 'tool_calls', None))
