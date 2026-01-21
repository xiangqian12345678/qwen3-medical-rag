"""
向量检索配置加载器 - 与索引构建完全对齐
"""
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .embed_config import (
    MilvusConfig, EmbedConfig, DenseFieldConfig,
    SparseFieldConfig, FusionConfig, DefaultSearchConfig
)

logger = logging.getLogger(__name__)


class EmbedConfigLoader:
    """向量检索配置加载器

    功能：
    - 从 YAML 文件加载向量检索配置
    - 支持环境变量替换
    - 与索引构建的配置结构完全对齐
    """

    _ENV_PATTERN = re.compile(r'\$\{([^}]+)\}')

    def __init__(self, config_path: str = None):
        """
        初始化向量检索配置加载器
        """
        if config_path is None:
            config_dir = Path(__file__).parent
            self.config_path = config_dir / "embed_config.yaml"
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # 替换环境变量
        raw = self._replace_env_vars(raw)

        self._dict = raw

        # 加载配置（与索引构建对齐）
        self.milvus = MilvusConfig(**raw.get("milvus", {}))
        self.dense_fields = {
            k: DenseFieldConfig(**v)
            for k, v in raw.get("dense_fields", {}).items()
        }
        self.sparse_fields = {
            k: SparseFieldConfig(**v)
            for k, v in raw.get("sparse_fields", {}).items()
        }
        self.fusion = FusionConfig(**raw.get("fusion", {}))
        self.default_search = DefaultSearchConfig(**raw.get("default_search", {}))

    def _replace_env_vars(self, data: Any) -> Any:
        """递归替换配置中的环境变量"""
        if isinstance(data, str):
            def replace_match(match):
                env_var = match.group(1)
                return os.getenv(env_var, match.group(0))
            return self._ENV_PATTERN.sub(replace_match, data)
        elif isinstance(data, dict):
            return {k: self._replace_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_env_vars(item) for item in data]
        else:
            return data

    @property
    def as_dict(self) -> dict:
        """返回当前配置的 dict 形式"""
        return self._dict

    def get(self, key: str, default=None):
        """获取配置项"""
        keys = key.split(".")
        value = self._dict
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def model_dump(self) -> dict:
        """返回配置的字典形式（兼容旧接口）"""
        return self.as_dict

    def save(self, save_path: str = None):
        """将当前配置保存到 YAML 文件"""
        path = save_path if save_path else self.config_path
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._dict, f, allow_unicode=True, sort_keys=False)
