"""配置包"""
try:
    # 尝试相对导入（当作为包导入时）
    from .models import *
except ImportError:
    # 回退到绝对导入（当直接运行时）
    from config.models import *

__all__ = [
    'AppConfig',
    'MilvusConfig',
    'DenseConfig',
    'SparseConfig',
    'EmbeddingConfig',
    'LLMConfig',
    'DataConfig',
    'AgentConfig',
    'MultiDialogueRagConfig',
    'SearchRequest',
    'SingleSearchRequest',
    'FusionSpec',
    'AnnsField',
    'OutputFields',
]
