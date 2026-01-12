# kb_factory.py
import os, atexit
from functools import lru_cache
from .KnowledgeBase import MedicalHybridKnowledgeBase
from ..config.models import AppConfig
import json

@lru_cache(maxsize=None)
def _kb_singleton(pid: int, config_str: str):
    """
    基于 (进程ID, 配置字符串) 的 KnowledgeBase 单例工厂函数

    设计目标：
    1. 在「单进程内」保证同一份配置只初始化一个 KB 实例
    2. 在「多进程」环境下（如 multiprocessing / gunicorn / uvicorn workers），
       不同进程各自维护自己的 KB 实例，避免跨进程资源污染
    3. 避免 KB 重复初始化带来的：
       - 向量库 / Milvus / ES 等重复连接
       - embedding / client 重复加载
       - 内存浪费与连接泄漏

    缓存策略：
    - 使用 functools.lru_cache
    - key = (pid, config_str)
    - maxsize=None 表示无限缓存（由进程生命周期兜底）

    参数：
    - pid (int):
        当前进程 ID，通过 os.getpid() 传入
        用于区分不同进程，防止跨进程复用对象
    - config_str (str):
        配置 dict 的 JSON 字符串表示（需稳定序列化）
        用于作为缓存 key，确保配置相同才能命中缓存

    返回：
    - MedicalHybridKnowledgeBase:
        已初始化完成、可直接使用的知识库实例
    """

    # 1. 将 JSON 字符串形式的配置反序列化为 dict
    #    注意：此处不直接接收 dict，是为了：
    #    - 保证 lru_cache 的 key 可 hash
    #    - 避免 dict 无序带来的 cache miss
    config_dict = json.loads(config_str)

    # 2. 使用 Pydantic / Schema 对配置进行校验与结构化
    #    - 提前发现配置错误
    #    - 保证下游 MedicalHybridKnowledgeBase 使用的是强类型配置
    config = AppConfig.model_validate(config_dict)

    # 3. 创建 Hybrid Knowledge Base 实例
    #    内部通常会完成：
    #    - 向量库 / 稀疏索引 / reranker client 初始化
    #    - embedding 模型加载
    #    - 连接池或长连接创建
    kb = MedicalHybridKnowledgeBase(config)

    # 4. 注册进程退出钩子，确保资源被优雅释放
    #    典型资源包括：
    #    - Milvus / ES / DB client
    #    - HTTP / GRPC 连接
    #
    #    getattr + 空 lambda 是为了：
    #    - client 不存在 close 方法时不报错
    #    - 保证 atexit 注册函数一定可调用
    atexit.register(
        lambda: getattr(kb.client, "close", lambda: None)()
    )

    # 5. 返回 KB 实例
    #    后续相同 (pid, config_str) 的调用将直接命中缓存
    return kb


def get_kb(config: dict):
    """
    对外暴露的 KB 获取接口（推荐唯一入口）

    设计目的：
    1. 对调用方隐藏：
       - 进程 ID
       - JSON 序列化细节
       - lru_cache 的存在
    2. 提供一个「幂等」的 KB 获取方式：
       - 同一进程 + 同一配置 -> 同一个 KB 实例
       - 不关心调用次数

    参数：
    - config (dict):
        原始配置字典（通常来自 yaml / env / 服务启动参数）

    返回：
    - MedicalHybridKnowledgeBase:
        当前进程内的 KB 单例
    """

    # 1. 获取当前进程 ID
    pid = os.getpid()

    # 2. 将配置 dict 序列化为稳定的 JSON 字符串
    #    sort_keys=True 的作用：
    #    - 保证相同内容的 dict 生成完全一致的字符串
    #    - 避免因为 key 顺序不同导致 cache miss
    config_str = json.dumps(config, sort_keys=True)

    # 3. 委托给带缓存的 singleton 工厂函数
    return _kb_singleton(pid, config_str)
