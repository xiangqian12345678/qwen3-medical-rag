"""向量检索工具函数"""
import json
import logging
import re
from typing import TYPE_CHECKING, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

from embed_config import DenseFieldConfig

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    pass


def create_embedding_client(config: Union[dict, DenseFieldConfig]) -> Embeddings:
    """创建嵌入客户端

    Args:
        config: 嵌入配置（字典或DenseFieldConfig对象），包含 provider, model, dimension, api_key, base_url

    Returns:
        嵌入客户端实例
    """
    if isinstance(config, dict):
        provider = config.get("provider")
        model = config.get("model")
        dimension = config.get("dimension")
        api_key = config.get("api_key")
        base_url = config.get("base_url")
    else:
        provider = config.provider
        model = config.model
        dimension = config.dimension
        api_key = config.api_key
        base_url = config.base_url

    if provider == "openai":
        kwargs = {
            "model": model,
            "dimensions": dimension
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIEmbeddings(**kwargs)

    elif provider == "ollama":
        kwargs = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbeddings(**kwargs)

    elif provider == "dashscope":
        kwargs = {
            "model": model
        }
        logger.info(f"创建 DashScope Embeddings: {model}")
        return DashScopeEmbeddings(**kwargs)

    else:
        raise ValueError(f"不支持的嵌入提供商: {provider}")


def json_to_list_document(text: str):
    """将JSON转换为Document列表"""
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


def _should_call_tool(last_ai: AIMessage) -> bool:
    """判断是否需要调用工具"""
    return bool(getattr(last_ai, 'tool_calls', None))


def strip_think_get_tokens(msg: AIMessage):
    """移除思考标签并获取token信息"""
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
        "msg": re.sub(r"```thinking\n.*?```", "", text, flags=re.DOTALL).strip(),
        "msg_len": msg_len,
        "msg_token_len": msg_token_len,
        "generate_time": dur
    }


def del_think(text: str) -> str:
    """移除思考过程标签"""
    return re.sub(r"```thinking\n.*?```", "", text, flags=re.DOTALL).strip()
