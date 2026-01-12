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
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，默认使用 simple_rag.yaml
        """
        if config_path is None:
            config_root = Path(__file__).parent.parent
            self.config_path = config_root / "simple_rag.yaml"
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
        """
        动态修改配置

        Args:
            updates: 变更内容
            save: 是否保存到文件
            save_path: 保存路径
        """
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
        """
        将点路径字典展开为嵌套字典

        该方法将使用点分隔符表示路径的扁平字典转换为嵌套字典结构。
        支持列表索引访问，格式为 "key[0].subkey"。

        Args:
            flat: 扁平字典，键为点分隔路径，值为要设置的值

        Returns:
            嵌套字典结构

        Examples:
            输入样例:
                flat = {
                    "model.name": "gpt-4",
                    "model.temperature": 0.7,
                    "model.api_keys[0]": "sk-xxx",
                    "model.api_keys[1]": "sk-yyy",
                    "database.host": "localhost",
                    "database.port": 5432
                }

            输出样例:
                {
                    "model": {
                        "name": "gpt-4",
                        "temperature": 0.7,
                        "api_keys": ["sk-xxx", "sk-yyy"]
                    },
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    }
                }

            列表索引示例:
                输入: {"servers[0].name": "web1", "servers[1].name": "web2"}
                输出: {"servers": [{"name": "web1"}, {"name": "web2"}]}

            混合路径示例:
                输入: {"a.b.c[0].x": 1, "a.b.c[0].y": 2, "a.b.c[1].x": 3}
                输出: {"a": {"b": {"c": [{"x": 1, "y": 2}, {"x": 3}]}}}
        """
        root = {}
        for key, value in flat.items():
            # 将点路径拆分为各个部分，例如 "model.name" -> ["model", "name"]
            parts = key.split(".")
            # cur 指向当前正在构建的嵌套层级
            cur = root

            for i, part in enumerate(parts):
                # 检查当前部分是否是列表索引格式，例如 "api_keys[0]"
                m = self._INDEX_PATTERN.match(part)
                if m:
                    # 提取列表名和索引，例如 "api_keys[0]" -> name="api_keys", idx=0
                    name, idx = m.group(1), int(m.group(2))

                    # 确保列表存在
                    if name not in cur or not isinstance(cur.get(name), list):
                        cur[name] = []
                    lst = cur[name]

                    # 如果列表长度不足，用空字典填充到指定索引位置
                    while len(lst) <= idx:
                        lst.append({})

                    # 如果是最后一个部分，直接赋值
                    if i == len(parts) - 1:
                        lst[idx] = value
                    # 否则继续向下构建嵌套结构
                    else:
                        if not isinstance(lst[idx], dict):
                            lst[idx] = {}
                        cur = lst[idx]
                else:
                    # 处理普通键（非列表索引）

                    # 如果是最后一个部分，直接赋值
                    if i == len(parts) - 1:
                        cur[part] = value
                    # 否则确保是字典类型并继续向下构建
                    else:
                        if part not in cur or not isinstance(cur[part], dict):
                            cur[part] = {}
                        cur = cur[part]
        return root

    def _deep_merge(self, base: Any, patch: Any) -> Any:
        """
        递归合并两个配置对象

        该方法将 patch 配置深度合并到 base 配置中，支持嵌套字典和列表的递归合并。
        对于同层级的同名键，会递归合并其值；对于列表，会逐个元素递归合并。

        Args:
            base: 基础配置（原配置）
            patch: 补丁配置（要合并的新配置）

        Returns:
            合并后的配置对象

        Examples:
            字典合并示例:
                base = {"model": {"name": "gpt-3", "temperature": 0.5}}
                patch = {"model": {"name": "gpt-4", "max_tokens": 2000}}
                输出: {"model": {"name": "gpt-4", "temperature": 0.5, "max_tokens": 2000}}

            列表合并示例:
                base = {"servers": [{"name": "web1", "port": 80}]}
                patch = {"servers": [{"port": 8080}, {"name": "web2"}]}
                输出: {"servers": [{"name": "web1", "port": 8080}, {"name": "web2"}]}

            嵌套结构示例:
                base = {"a": {"b": {"c": 1, "d": 2}}}
                patch = {"a": {"b": {"c": 10, "e": 3}}}
                输出: {"a": {"b": {"c": 10, "d": 2, "e": 3}}}

            类型覆盖示例:
                base = {"count": 5}
                patch = {"count": 10}
                输出: {"count": 10}
        """
        if isinstance(base, dict) and isinstance(patch, dict):
            # 字典合并：创建 base 的副本
            out = dict(base)
            for k, v in patch.items():
                if k in out:
                    # 键已存在，递归合并其值
                    out[k] = self._deep_merge(out[k], v)
                else:
                    # 键不存在，直接添加
                    out[k] = v
            return out
        elif isinstance(base, list) and isinstance(patch, list):
            # 列表合并：创建 base 的副本
            out = list(base)
            for i, v in enumerate(patch):
                if i < len(out):
                    # 索引有效，递归合并对应位置的元素
                    out[i] = self._deep_merge(out[i], v)
                else:
                    # 超出原列表长度，直接追加新元素
                    out.append(v)
            return out
        else:
            # 非容器类型（字符串、数字等），直接用 patch 值覆盖
            return patch
