"""配置加载器"""
import logging
import re
from pathlib import Path
from typing import Union, List, Tuple, Any
import yaml

from .models import AppConfig


logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器"""

    _INDEX_PATTERN = re.compile(r"(.*?)\[(\d+)\]$")

    def __init__(self, config_path: str = None):
        """初始化配置加载器"""
        if config_path is None:
            config_root = Path(__file__).parent.parent
            self.config_path = config_root / "single_dialogue_agent.yaml"
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        self._dict = raw
        self._config = AppConfig(**raw)

    @property
    def config(self) -> AppConfig:
        """获取当前配置"""
        return self._config

    @property
    def as_dict(self) -> dict:
        """返回配置的字典形式"""
        return self._config.model_dump()

    def change(
        self,
        updates: Union[dict, List[Tuple[str, Any]]],
        save: bool = False,
        save_path: str = ""
    ) -> AppConfig:
        """动态修改配置"""
        if isinstance(updates, dict):
            upd_dict = self._expand_dot_paths(updates)
        else:
            upd_dict = self._expand_dot_paths(dict(updates))

        merged = self._deep_merge(self._dict, upd_dict)
        new_config = AppConfig(**merged)

        self._dict = merged
        self._config = new_config

        if save:
            path = save_path if save_path else self.config_path
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._dict, f, allow_unicode=True, sort_keys=False)

        return self._config

    def _expand_dot_paths(self, flat: dict) -> dict:
        """将点路径字典展开为嵌套字典"""
        root = {}
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
