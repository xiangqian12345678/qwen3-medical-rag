"""检索器 - 支持混合检索和向量融合"""
import logging
from typing import List, Dict, Any, Optional

from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

from config import IndexConfig, SearchRequest, SingleSearchRequest, FusionSpec
from vectorizer import VectorFieldProcessor

logger = logging.getLogger(__name__)


class KnowledgeBaseSearcher:
    """知识库检索器"""

    def __init__(self, config: IndexConfig, client):
        """初始化检索器

        Args:
            config: 索引配置
            client: Milvus 客户端
        """
        self.config = config
        self.client = client
        self.field_processor = VectorFieldProcessor(config)

    def search(self, request: SearchRequest) -> List[Dict[str, Any]]:
        """执行混合检索

        Args:
            request: 检索请求

        Returns:
            List[Dict[str, Any]]: 检索结果
        """
        collection_name = request.collection_name

        # 构建 AnnSearchRequest 列表
        ann_search_requests = []
        for single_req in request.requests:
            ann_search_requests.append(
                self._build_ann_search_request(request.query, single_req)
            )

        # 如果只有一个请求且不需要融合，直接搜索
        if len(ann_search_requests) == 1 and request.fuse is None:
            req = ann_search_requests[0]
            search_params = {
                "metric_type": req.param["metric_type"],
                "params": req.param["params"]
            }
            results = self.client.search(
                collection_name=collection_name,
                data=req.data,
                anns_field=req.anns_field,
                search_params=search_params,
                limit=req.limit,
                output_fields=request.output_fields
            )
        else:
            # 多路检索或需要融合
            ranker = self._create_ranker(request.fuse)
            results = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=ann_search_requests,
                ranker=ranker,
                limit=request.top_k,
                output_fields=request.output_fields
            )

        # 展平结果
        flattened = []
        for result_batch in results:
            for item in result_batch:
                item_dict = item.to_dict()
                flattened.append(item_dict)

        # 按 limit 截断
        return flattened[:request.limit]

    def _build_ann_search_request(self, query: str, 
                                  single_req: SingleSearchRequest) -> AnnSearchRequest:
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
            # 稀疏向量 - 返回原始文本（Milvus BM25）或稀疏向量
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

    def _create_ranker(self, fusion: Optional[FusionSpec]):
        """创建融合器

        Args:
            fusion: 融合配置

        Returns:
            RRFRanker 或 WeightedRanker
        """
        if fusion is None:
            return RRFRanker(k=60)

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

    def deduplicate_by_origin_pk(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 origin_pk 去重，保留第一个结果

        Args:
            results: 检索结果列表

        Returns:
            List[Dict[str, Any]]: 去重后的结果
        """
        seen = set()
        deduplicated = []

        for item in results:
            origin_pk = item.get("entity", {}).get("origin_pk")
            if origin_pk and origin_pk not in seen:
                seen.add(origin_pk)
                deduplicated.append(item)

        return deduplicated
