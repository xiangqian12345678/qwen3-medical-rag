"""向量检索器 - 支持混合检索和向量融合"""
import logging
from typing import Dict, Any, Optional, Union, List

from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker, WeightedRanker
from langchain_core.documents import Document

from .embed_config import (
    EmbedConfig, SearchRequest, SingleSearchRequest, FusionSpec,
    DenseFieldConfig, SparseFieldConfig
)
from .embed_loader import EmbedConfigLoader
from .embedding_client import create_embedding_client
from .sparse_vectorizer import SparseVectorProcessor
from .embed_vocab import Vocabulary

logger = logging.getLogger(__name__)

# 知识库实例缓存
_kb_instance: Optional["EmbedSearcher"] = None


class EmbedSearcher:
    """知识库检索器，封装向量数据库检索功能"""

    def __init__(self, config: Union[EmbedConfigLoader, Dict[str, Any], EmbedConfig]):
        """
        初始化知识库

        Args:
            config: 应用配置（EmbedConfigLoader实例、配置字典或EmbedConfig实例）
        """
        # 如果传入的是字典，转换为EmbedConfig对象
        if isinstance(config, dict):
            self.config = EmbedConfig(**config)
        elif isinstance(config, EmbedConfigLoader):
            self.config = EmbedConfig(
                milvus=config.milvus,
                dense_fields=config.dense_fields,
                sparse_fields=config.sparse_fields,
                fusion=config.fusion,
                default_search=config.default_search
            )
        else:
            # 传入的是EmbedConfig实例
            self.config = config

        self.collection_name = self.config.milvus.collection_name

        # 连接Milvus
        logger.info(f"连接Milvus: {self.config.milvus.uri}")
        connections.connect(
            alias="default",
            uri=self.config.milvus.uri,
            token=self.config.milvus.token
        )

        # 获取集合
        self.collection = Collection(self.collection_name)
        logger.info(f"加载集合: {self.collection_name}")

        # 初始化向量处理器
        self.field_processor = VectorFieldProcessor(self.config)

    def search(self, req: SearchRequest) -> List[Document]:
        """
        混合检索：多路向量检索 + 融合

        Args:
            req: 检索请求对象

        Returns:
            List[Document]: 检索结果列表
        """
        collection_name = req.collection_name or self.collection_name

        # 构建 AnnSearchRequest 列表
        ann_search_requests = []
        for single_req in req.requests:
            ann_search_requests.append(
                self._build_ann_search_request(req.query, single_req)
            )

        # 如果只有一个请求且不需要融合，直接搜索
        if len(ann_search_requests) == 1 and req.fuse is None:
            req_obj = ann_search_requests[0]
            search_params = {
                "metric_type": req_obj.param["metric_type"],
                "params": req_obj.param["params"]
            }
            results = self.collection.search(
                data=req_obj.data,
                anns_field=req_obj.anns_field,
                param=search_params,
                limit=req_obj.limit,
                expr=req_obj.expr,
                output_fields=req.output_fields or self.config.default_search.output_fields
            )
        else:
            # 多路检索或需要融合
            ranker = self._create_ranker(req.fuse or self.config.fusion)
            results = self.collection.hybrid_search(
                reqs=ann_search_requests,
                rerank=ranker,
                limit=req.top_k or self.config.default_search.top_k,
                output_fields=req.output_fields or self.config.default_search.output_fields
            )

        # 转换为Document对象
        documents = []
        for hit in results[0]:
            metadata = {k: v for k, v in hit.entity.items() if k != "chunk" and k != "summary"}
            content = hit.entity.get("chunk") or hit.entity.get("summary") or ""
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        # 按 limit 截断
        limit = req.limit or self.config.default_search.limit
        return documents[:limit]

    def _build_ann_search_request(self, query: str, single_req: SingleSearchRequest) -> AnnSearchRequest:
        """构建 AnnSearchRequest 子请求对象

        Args:
            query: 查询文本
            single_req: 单个检索请求配置

        Returns:
            AnnSearchRequest: Milvus 检索请求对象
        """
        # 编码查询数据
        encoded_data = self._encode_query(query, single_req.anns_field)

        return AnnSearchRequest(
            data=[encoded_data],
            anns_field=single_req.anns_field,
            param={
                "metric_type": single_req.metric_type,
                "params": single_req.search_params
            },
            limit=single_req.limit,
            expr=single_req.expr
        )

    def _encode_query(self, query: str, anns_field: str) -> Any:
        """编码查询数据

        Args:
            query: 查询文本
            anns_field: 检索字段名称

        Returns:
            Any: 编码后的查询数据（向量或原始文本）
        """
        # 判断是稠密向量还是稀疏向量
        if anns_field.endswith("_sparse"):
            # 稀疏向量
            for field_name, field_config in self.config.sparse_fields.items():
                if field_config.index_field == anns_field:
                    return self.field_processor.get_sparse_vector(query, field_name)
            return query
        else:
            # 稠密向量
            for field_name, field_config in self.config.dense_fields.items():
                if field_config.index_field == anns_field:
                    return self.field_processor.get_dense_embedding(field_name, query)
            raise ValueError(f"未知的检索字段: {anns_field}")

    def _create_ranker(self, fusion: FusionSpec):
        """创建融合器

        Args:
            fusion: 融合配置

        Returns:
            RRFRanker 或 WeightedRanker
        """
        if fusion.method == "rrf":
            return RRFRanker(k=fusion.k)
        elif fusion.method == "weighted":
            if fusion.weights is None:
                # 使用配置中的权重
                weights = list(self.config.fusion.weights.values())
            else:
                weights = list(fusion.weights.values())
            return WeightedRanker(weights)
        else:
            return RRFRanker(k=60)

    def __del__(self):
        """析构函数，关闭连接"""
        try:
            from pymilvus import connections
            connections.disconnect("default")
        except:
            pass


class VectorFieldProcessor:
    """向量字段处理器 - 处理稠密和稀疏向量"""

    def __init__(self, config: EmbedConfig):
        """初始化向量字段处理器

        Args:
            config: 嵌入配置对象，包含稠密和稀疏字段的配置信息
        """
        self.config = config
        # 稠密向量嵌入器字典：key为字段名，value为嵌入器实例
        self.dense_embedders: Dict[str, Any] = {}
        # 稀疏向量处理器（BM25）：全库共享一个词表和处理器
        self.sparse_processor: Optional[SparseVectorProcessor] = None

        # 初始化稠密向量嵌入器
        for field_name, field_config in config.dense_fields.items():
            if field_config.embed:
                self.dense_embedders[field_name] = create_embedding_client(field_config)

        # 初始化稀疏向量处理器（BM25）
        for field_name, field_config in config.sparse_fields.items():
            if field_config.embed:
                vocab = Vocabulary.load(field_config.vocab_path)
                if vocab is None:
                    raise FileNotFoundError(f"词表加载失败: {field_config.vocab_path}")
                self.sparse_processor = SparseVectorProcessor(vocab, field_config)

    def get_dense_embedding(self, field_name: str, text: str) -> List[float]:
        """获取稠密向量嵌入

        Args:
            field_name: 字段名称
            text: 输入文本

        Returns:
            List[float]: 向量嵌入
        """
        if field_name not in self.dense_embedders:
            raise ValueError(f"字段 {field_name} 未配置稠密向量")
        return self.dense_embedders[field_name].embed_query(text)

    def get_sparse_vector(self, text: str, field_name: str = "chunk") -> Dict[int, float]:
        """获取稀疏向量

        Args:
            text: 输入文本
            field_name: 字段名称（用于获取配置）

        Returns:
            Dict[int, float]: 稀疏向量
        """
        if self.sparse_processor is None:
            raise ValueError("未配置稀疏向量处理器")
        if field_name not in self.config.sparse_fields:
            raise ValueError(f"字段 {field_name} 未配置稀疏向量")

        field_config = self.config.sparse_fields[field_name]
        avgdl = self.sparse_processor.vocab.avgdl if self.sparse_processor.vocab.N > 0 else 100.0
        return self.sparse_processor.build_sparse_vector(text, avgdl)


def get_kb(config: Dict[str, Any] = None) -> EmbedSearcher:
    """
    获取知识库实例（单例模式）

    Args:
        config: 配置字典

    Returns:
        EmbedSearcher: 知识库实例
    """
    global _kb_instance

    if _kb_instance is None:
        if config is None:
            raise ValueError("首次调用必须传入config参数")

        _kb_instance = EmbedSearcher(config)

    return _kb_instance


def reset_kb():
    """重置知识库实例（用于测试或重新初始化）"""
    global _kb_instance
    _kb_instance = None
