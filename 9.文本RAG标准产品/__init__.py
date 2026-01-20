"""Agent包"""

# 使用相对导入确保引用当前子项目的模块
try:
    # 尝试相对导入（当作为包导入时）
    from .multi_dialogue_agent import MultiDialogueAgent, MedicalAgentState
    # SearchGraph 在 MultiDialogueAgent 中使用，延迟导入避免循环依赖
    # from .search_graph import SearchGraph, SearchMessagesState
    from .utils import (
        strip_think_get_tokens,
        del_think,
        json_to_list_document,
        format_document_str,
        _should_call_tool,
        create_llm_client,
        create_embedding_client,
        format_documents,
    )
    from .milvus.embed_searcher import get_kb, EmbedSearcher, reset_kb
    from .search.web_search import NetworkSearchResult
except ImportError:
    # 回退到绝对导入（当直接运行文件时）
    from multi_dialogue_agent import MultiDialogueAgent, MedicalAgentState
    # SearchGraph 在 MultiDialogueAgent 中使用，延迟导入避免循环依赖
    # from search_graph import SearchGraph, SearchMessagesState
    from utils import (
        strip_think_get_tokens,
        del_think,
        json_to_list_document,
        format_document_str,
        _should_call_tool,
        create_llm_client,
        create_embedding_client,
        format_documents,
    )
    from milvus.embed_searcher import get_kb, EmbedSearcher, reset_kb
    from search.web_search import NetworkSearchResult

# 提供延迟导入函数
def _import_search_graph():
    """延迟导入SearchGraph和SearchMessagesState"""
    try:
        from .search_graph import SearchGraph, SearchMessagesState
    except ImportError:
        from search_graph import SearchGraph, SearchMessagesState
    return SearchGraph, SearchMessagesState


# 使用__getattr__实现延迟导入
def __getattr__(name: str):
    if name in ('SearchGraph', 'SearchMessagesState'):
        SearchGraph, SearchMessagesState = _import_search_graph()
        globals()[name] = SearchGraph if name == 'SearchGraph' else SearchMessagesState
        return globals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    'MultiDialogueAgent',
    'MedicalAgentState',
    'SearchGraph',
    'SearchMessagesState',
    'NetworkSearchResult',
    'strip_think_get_tokens',
    'del_think',
    'json_to_list_document',
    'format_document_str',
    '_should_call_tool',
    'create_llm_client',
    'create_embedding_client',
    'format_documents',
    'get_kb',
    'EmbedSearcher',
    'reset_kb',
]
