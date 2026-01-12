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
    - 支持列表下标访问（如 "items[0].name"）
    - 使用 Pydantic 进行配置校验
    - 可选择将修改持久化到 YAML 文件
    """

    _INDEX_PATTERN = re.compile(r"(.*?)\[(\d+)\]$")  # 用于解析 a.b[0].c 格式的路径

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器
        Args:
            config_path: 配置文件路径（不是目录）。默认使用当前文件同目录下的 app_config.yaml
        Raises:
            FileNotFoundError: 当配置文件不存在时抛出
        """
        if config_path is None:
            config_root = Path(__file__).parent
            self.config_path = config_root / "app_config.yaml"
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        self._dict = raw  # 保存原始 dict，用于深合并
        self._app_config = AppConfig(**raw)  # 使用 Pydantic 校验配置

    @property
    def config(self) -> AppConfig:
        """获取当前配置的 Pydantic 模型实例"""
        return self._app_config

    @property
    def as_dict(self) -> dict:
        """返回当前配置的 dict 形式（深拷贝）"""
        return self._app_config.model_dump()

    # -------------------------------------------------------------------------
    # 公共方法：change
    # -------------------------------------------------------------------------
    def change(
            self,
            updates: Union[dict, List[tuple[str, Any]]],
            save: bool = False,
            save_path: str = ""
    ) -> AppConfig:
        """动态修改配置的任意字段

        支持两种更新形式：
          1) 嵌套 dict：{"llm": {"model": "qwen3:72b"}}
          2) 点路径：{"embedding.text_dense.model": "bge-m3:latest"}
             也支持列表下标： "foo.bar[0].baz": 123

        Args:
            updates: 变更内容，支持 dict 或 [(path, value), ...] 格式
            save: 是否立即写回 YAML 文件，默认 False
            save_path: 保存路径，为空时使用原配置文件路径

        Returns:
            更新并校验后的 AppConfig 实例

        Examples:
            >>> loader.change({"llm.model": "qwen3:72b"})
            >>> loader.change([("embedding.text_dense.model", "bge-m3:latest")])
        """
        # 把点路径更新转成嵌套 dict
        if isinstance(updates, dict):
            upd_dict = self._expand_dot_paths(updates)
        else:
            # 支持传入 [("a.b", 1), ...] 格式
            upd_dict = self._expand_dot_paths(dict(updates))

        # 深合并到现有 dict，保留未修改的字段
        merged = self._deep_merge(self._dict, upd_dict)

        # 用 Pydantic 校验新配置
        new_config = AppConfig(**merged)

        # 更新内部状态
        self._dict = merged
        self._app_config = new_config

        # 如果需要保存，写回 YAML 文件
        if save:
            self._save_yaml(save_path if save_path else self.config_path)

        return self._app_config

    # -------------------------------------------------------------------------
    # 私有方法
    # -------------------------------------------------------------------------
    def _expand_dot_paths(self, flat: dict) -> dict:
        """将点路径形式的字典展开为嵌套字典

        将 {"a.b[0].c": 1, "x.y": 2} 展开为 {"a": {"b": [{"c": 1}]}, "x": {"y": 2}}

        Args:
            flat: 扁平化的点路径字典

        Returns:
            嵌套结构的字典

        Examples:
            >>> _expand_dot_paths({"a.b.c": 1})
            {'a': {'b': {'c': 1}}}
            >>> _expand_dot_paths({"items[0].name": "foo"})
            {'items': [{'name': 'foo'}]}
        """
        root: dict = {}  # 初始化根字典，用于存储展开后的嵌套结构
        for key, value in flat.items():  # 遍历扁平字典中的每个键值对
            parts = key.split(".")  # 将点路径按 "." 分割成多个部分，如 "a.b.c" -> ["a", "b", "c"]
            cur = root  # cur 指向当前正在处理的字典节点，初始时指向根字典
            for i, part in enumerate(parts):  # 遍历路径的每一部分，i 为索引，part 为当前部分
                m = self._INDEX_PATTERN.match(part)  # 使用正则表达式匹配是否包含数组下标，如 "items[0]"
                if m:
                    # 处理带下标的部分，如 "items[0]"
                    name, idx = m.group(1), int(m.group(2))  # 提取数组名称（如 "items"）和索引（如 0）
                    if name not in cur or not isinstance(cur.get(name), list):
                        # 如果当前字典中不存在该名称，或者该名称对应的值不是列表，则初始化为空列表
                        cur[name] = []
                    lst = cur[name]  # 获取对应的列表
                    # 确保列表长度足够，通过填充空字典来扩充列表
                    while len(lst) <= idx:
                        lst.append({})
                    if i == len(parts) - 1:
                        # 最后一个部分，直接设置值到列表的指定索引位置
                        lst[idx] = value
                    else:
                        # 中间部分，需要继续展开，确保当前位置是字典类型以便继续嵌套
                        # 例如： flat = {"users[0].address.city": "北京"}
                        # 最终：
                        #     {
                        #         "users": [
                        #             {
                        #                 "address": {
                        #                     "city": "北京"
                        #                 }
                        #             }
                        #         ]
                        #     }
                        if not isinstance(lst[idx], dict):
                            lst[idx] = {}
                        cur = lst[idx]  # 将 cur 指向列表指定索引位置的字典，准备处理下一级路径
                else:
                    # 处理不带下标的普通部分
                    if i == len(parts) - 1:
                        # 最后一个部分，直接设置值到字典中
                        cur[part] = value
                    else:
                        # 中间部分，需要创建或进入嵌套字典
                        if part not in cur or not isinstance(cur[part], dict):
                            # 如果不存在或不是字典类型，则初始化为空字典
                            cur[part] = {}
                        cur = cur[part]  # 将 cur 指向下一级字典，准备处理下一级路径
        return root  # 返回展开后的嵌套字典结构

    def _deep_merge(self, base: Any, patch: Any) -> Any:
        """递归合并两个配置对象

        合并规则：
        - dict: 深度合并，相同 key 递归合并
        - list: 按索引覆盖，patch 长度优先生效
        - 其他类型: 直接用 patch 替换 base

        Args:
            base: 原始配置对象
            patch: 要应用的更新配置对象

        Returns:
            合并后的配置对象

        Examples:
            >>> _deep_merge({"a": {"x": 1}}, {"a": {"y": 2}})
            {'a': {'x': 1, 'y': 2}}
            >>> _deep_merge([1, 2, 3], [10, 20])
            [10, 20, 3]
        """
        if isinstance(base, dict) and isinstance(patch, dict):
            # 字典：递归合并每个 key
            out = dict(base)
            for k, v in patch.items():
                if k in out:
                    out[k] = self._deep_merge(out[k], v)
                else:
                    out[k] = v
            return out
        elif isinstance(base, list) and isinstance(patch, list):
            # 列表：按索引覆盖
            out = list(base)
            for i, v in enumerate(patch):
                if i < len(out):
                    out[i] = self._deep_merge(out[i], v)
                else:
                    out.append(v)
            return out
        else:
            # 其他类型：直接用 patch 替换
            return patch

    def _save_yaml(self, save_path):
        """将当前配置保存到 YAML 文件

        Args:
            save_path: 目标文件路径

        Note:
            使用 safe_dump 确保安全性
            allow_unicode=True 支持中文字符
            sort_keys=False 保持原始顺序
        """
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._dict, f, allow_unicode=True, sort_keys=False)
