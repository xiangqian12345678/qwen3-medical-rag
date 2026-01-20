"""数据库工厂类，用于创建和管理知识库实例"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 添加当前项目目录到 Python 路径（支持直接运行）
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# 导入配置和工具函数
try:
    # 尝试相对导入（当作为包导入时）
    from ..config.models import AppConfig, SearchRequest
    from ..agent.utils import create_embedding_client
except ImportError:
    # 回退到绝对导入（当直接运行文件时）
    from config.models import AppConfig, SearchRequest
    from agent.utils import create_embedding_client

logger = logging.getLogger(__name__)

# 知识库实例缓存
_kb_instance: Optional["EmbedSearcher"] = None


class EmbedSearcher:
    """知识库类，封装向量数据库检索功能"""

    def __init__(self, config: AppConfig):
        """
        初始化知识库

        Args:
            config: 应用配置
        """
        from pymilvus import connections, Collection

        self.config = config
        self.collection_name = config.milvus.collection_name

        # 连接Milvus
        logger.info(f"连接Milvus: {config.milvus.uri}")
        connections.connect(
            alias="default",
            uri=config.milvus.uri,
            token=config.milvus.token
        )

        # 获取集合
        self.collection = Collection(self.collection_name)
        logger.info(f"加载集合: {self.collection_name}")

    def search(self, req: SearchRequest):
        """
        混合检索：多路向量检索 + 融合

        Args:
            req: 检索请求对象

        Returns:
            检索结果列表
        """
        from pymilvus import AnnSearchRequest, RRFRanker
        from langchain_core.documents import Document

        # 如果没有指定检索请求，使用默认配置（多路检索）
        if not req.requests:
            req.requests = [
                SingleSearchRequest(
                    anns_field="chunk_dense",
                    limit=50,
                    search_params={"ef": 64}
                ),
                SingleSearchRequest(
                    anns_field="questions_dense",
                    limit=50,
                    search_params={"ef": 64}
                ),
                SingleSearchRequest(
                    anns_field="parent_chunk_dense",
                    limit=50,
                    search_params={"ef": 64}
                )
            ]

        # 构建检索请求
        search_requests = []
        for r in req.requests:
            # 生成查询向量
            if r.anns_field in ["chunk_dense", "parent_chunk_dense", "questions_dense"]:
                embedding_config = self.config.embedding.text_dense
                embedding_client = create_embedding_client(embedding_config)
                query_vector = embedding_client.embed_query(req.query)
            else:
                # 稀疏向量，这里使用简单的BM25模拟
                query_vector = []

            # 构建检索请求
            ann_req = AnnSearchRequest(
                data=[query_vector],
                anns_field=r.anns_field,
                param=r.search_params,
                limit=r.limit,
                expr=r.expr
            )
            search_requests.append(ann_req)

        # 执行检索
        if len(search_requests) == 1:
            # 单路检索
            results = self.collection.search(
                data=search_requests[0].data,
                anns_field=search_requests[0].anns_field,
                param=search_requests[0].param,
                limit=search_requests[0].limit,
                expr=search_requests[0].expr,
                output_fields=req.output_fields
            )
        else:
            # 多路检索 + 融合
            if req.fuse.method == "rrf":
                ranker = RRFRanker(k=req.fuse.k)
            else:
                # 加权融合
                ranker = None
                # TODO: 实现加权融合
                logger.warning("加权融合暂未实现，使用RRF")
                ranker = RRFRanker(k=60)

            results = self.collection.hybrid_search(
                reqs=search_requests,
                rerank=ranker,
                limit=req.limit,
                output_fields=req.output_fields
            )

        # 转换为Document对象
        documents = []
        for hit in results[0]:
            metadata = {k: v for k, v in hit.entity.items() if k != "chunk" and k != "summary"}
            content = hit.entity.get("chunk") or hit.entity.get("summary") or ""
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def __del__(self):
        """析构函数，关闭连接"""
        try:
            from pymilvus import connections
            connections.disconnect("default")
        except:
            pass


def get_kb(config: Dict[str, Any] = None) -> EmbedSearcher:
    """
    获取知识库实例（单例模式）

    Args:
        config: 配置字典

    Returns:
        知识库实例
    """
    global _kb_instance

    if _kb_instance is None:
        if config is None:
            raise ValueError("首次调用必须传入config参数")

        from config.models import AppConfig
        app_config = AppConfig(**config) if isinstance(config, dict) else config
        _kb_instance = EmbedSearcher(app_config)

    return _kb_instance


def reset_kb():
    """重置知识库实例（用于测试或重新初始化）"""
    global _kb_instance
    _kb_instance = None
