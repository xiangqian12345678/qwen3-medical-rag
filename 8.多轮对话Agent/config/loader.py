"""
配置加载器
"""
import logging
import re
from pathlib import Path
from typing import List, Optional, Any, Union

import yaml

from .models import AppConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器

    功能：
    - 从 YAML 文件加载配置
    - 支持通过点路径或嵌套 dict 动态修改配置
    - 支持列表下标访问
    - 使用 Pydantic 进行配置校验
    """

    _INDEX_PATTERN = re.compile(r"(.*?)\[(\d+)\]$")

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器
        """
        if config_path is None:
            config_root = Path(__file__).parent.parent
            self.config_path = config_root / "multi_dialogue_agent.yaml"
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        self._dict = raw
        self._app_config = AppConfig(**raw)

    @property
    def config(self) -> AppConfig:
        """获取当前配置的 Pydantic 模型实例"""
        return self._app_config

    @property
    def as_dict(self) -> dict:
        """返回当前配置的 dict 形式（深拷贝）"""
        return self._app_config.model_dump()

    def change(
            self,
            updates: Union[dict, List[tuple[str, Any]]],
            save: bool = False,
            save_path: str = ""
    ) -> AppConfig:
        """动态修改配置的任意字段"""
        if isinstance(updates, dict):
            upd_dict = self._expand_dot_paths(updates)
        else:
            upd_dict = self._expand_dot_paths(dict(updates))

        merged = self._deep_merge(self._dict, upd_dict)
        new_config = AppConfig(**merged)

        self._dict = merged
        self._app_config = new_config

        if save:
            self._save_yaml(save_path if save_path else self.config_path)

        return self._app_config

    def _expand_dot_paths(self, flat: dict) -> dict:
        """将点路径形式的字典展开为嵌套字典"""
        root: dict = {}
        for key, value in flat.items():
            parts = key.split(".")
            cur = root
            for i, part in enumerate(parts):
                m = self._INDEX_PATTERN.match(part)
                if m:
                    name, idx = m.group(1), int(m.group(2))
                    if name not in cur or not isinstance(cur.get(name), list):
                        cur[name] = []
                    lst = cur[name]
                    while len(lst) <= idx:
                        lst.append({})
                    if i == len(parts) - 1:
                        lst[idx] = value
                    else:
                        if not isinstance(lst[idx], dict):
                            lst[idx] = {}
                        cur = lst[idx]
                else:
                    if i == len(parts) - 1:
                        cur[part] = value
                    else:
                        if part not in cur or not isinstance(cur[part], dict):
                            cur[part] = {}
                        cur = cur[part]
        return root

    def _deep_merge(self, base: Any, patch: Any) -> Any:
        """递归合并两个配置对象"""
        if isinstance(base, dict) and isinstance(patch, dict):
            out = dict(base)
            for k, v in patch.items():
                if k in out:
                    out[k] = self._deep_merge(out[k], v)
                else:
                    out[k] = v
            return out
        elif isinstance(base, list) and isinstance(patch, list):
            out = list(base)
            for i, v in enumerate(patch):
                if i < len(out):
                    out[i] = self._deep_merge(out[i], v)
                else:
                    out.append(v)
            return out
        else:
            return patch

    def _save_yaml(self, save_path):
        """将当前配置保存到 YAML 文件"""
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._dict, f, allow_unicode=True, sort_keys=False)
