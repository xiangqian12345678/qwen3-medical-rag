"""知识库工厂"""
import logging
import hashlib
from typing import List, Any, Dict
from langchain_core.documents import Document
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker

from config.models import AppConfig, SearchRequest, SingleSearchRequest
from .utils import create_embedding_client

logger = logging.getLogger(__name__)


# 进程级单例缓存
_kb_cache = {}


def get_kb(config_dict: dict) -> "KnowledgeBase":
    """获取知识库单例"""
    cache_key = hashlib.md5(str(config_dict).encode()).hexdigest()
    if cache_key not in _kb_cache:
        _kb_cache[cache_key] = KnowledgeBase(AppConfig(**config_dict))
    return _kb_cache[cache_key]


class KnowledgeBase:
    """知识库"""

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.milvus_config = app_config.milvus
        self.embedding_config = app_config.embedding
        self.data_config = app_config.data

        # 创建Milvus客户端
        self.client = MilvusClient(
            uri=self.milvus_config.uri,
            token=self.milvus_config.token
        )

        # 嵌入器字典
        self.embedders = {}

        # 初始化chunk_dense嵌入器
        self.embedders["chunk_dense"] = create_embedding_client(
            self.embedding_config.summary_dense
        )

        # 初始化parent_chunk_dense嵌入器
        self.embedders["parent_chunk_dense"] = create_embedding_client(
            self.embedding_config.summary_dense
        )

        # 初始化questions_dense嵌入器
        self.embedders["questions_dense"] = create_embedding_client(
            self.embedding_config.summary_dense
        )

        # 初始化text_sparse（BM25）
        sparse_config = self.embedding_config.text_sparse
        if sparse_config.provider == "self":
            # BM25稀疏向量需要从外部词表加载
            # 这里简化处理，实际使用时需要实现BM25SparseEmbedding
            pass

        logger.info(f"知识库初始化完成，集合: {self.milvus_config.collection_name}")

    def _encode_query(self, query: str, anns_field: str) -> Any:
        """编码查询"""
        if anns_field in self.embedders:
            return self.embedders[anns_field].embed_query(query)
        else:
            # 对于text_sparse，直接返回文本
            return query

    def _build_ann_search_request(
            self,
            query: str,
            single_request: SingleSearchRequest
    ) -> AnnSearchRequest:
        """构建AnnSearchRequest"""
        encoded_data = self._encode_query(query, single_request.anns_field)

        return AnnSearchRequest(
            data=[encoded_data],
            anns_field=single_request.anns_field,
            param={
                "metric_type": single_request.metric_type,
                "params": single_request.search_params
            },
            limit=single_request.limit,
            expr=single_request.expr
        )

    def _hybrid_search(self, req: SearchRequest) -> List[Dict]:
        """混合检索"""
        # 构建子请求
        anns = []
        for single_req in req.requests:
            anns.append(self._build_ann_search_request(req.query, single_req))

        # 创建融合器
        if req.fuse.method == "rrf":
            ranker = RRFRanker(k=req.fuse.k)
        else:
            if req.fuse.weights:
                weight_list = req.fuse.weights if isinstance(req.fuse.weights, list) else list(req.fuse.weights.values())
            else:
                weight_list = [0.4, 0.4, 0.2]
            ranker = WeightedRanker(*weight_list)

        # 执行混合检索
        result = self.client.hybrid_search(
            collection_name=req.collection_name,
            reqs=anns,
            ranker=ranker,
            limit=req.top_k if hasattr(req, 'top_k') else req.limit,
            output_fields=req.output_fields
        )

        return result[0] if result else []

    def search(self, req: SearchRequest) -> List[Document]:
        """执行知识库检索"""
        if len(req.requests) == 1:
            # 单路检索
            data = self._encode_query(req.query, req.requests[0].anns_field)
            outputs = self.client.search(
                collection_name=req.collection_name,
                data=[data],
                filter=req.requests[0].expr,
                limit=req.requests[0].limit,
                output_fields=req.output_fields,
                search_params={
                    "metric_type": req.requests[0].metric_type,
                    "params": req.requests[0].search_params
                },
                anns_field=req.requests[0].anns_field
            )
            outputs = outputs[0] if outputs else []
        else:
            # 混合检索
            outputs = self._hybrid_search(req)

        # 转换为Document对象
        results = []
        for item in outputs[:req.limit]:
            # 处理Milvus返回的对象格式
            if hasattr(item, 'to_dict'):
                item_dict = item.to_dict()
                entity = item_dict.get("entity", {})
                distance = item_dict.get("distance", 99999)
            elif isinstance(item, dict):
                entity = item.get("entity", {})
                distance = item.get("distance", 99999)
            else:
                entity = {}
                distance = 99999

            # 优先使用 chunk 字段作为主要内容，如果没有则使用 summary
            text = entity.get(self.data_config.chunk_field, "")
            if not text:
                text = entity.get(self.data_config.summary_field, "")

            # 追加 parent_chunk（如果有）
            parent_chunk = entity.get(self.data_config.parent_chunk_field, "")
            if parent_chunk and parent_chunk != text:
                text = f"{text}\n\n{parent_chunk}"

            results.append(Document(
                page_content=text,
                metadata={
                    "pk": entity.get("pk", ""),
                    "distance": distance,
                    "chunk": entity.get(self.data_config.chunk_field, ""),
                    "parent_chunk": entity.get(self.data_config.parent_chunk_field, ""),
                    "summary": entity.get(self.data_config.summary_field, ""),
                    "questions": entity.get(self.data_config.questions_field, ""),
                    "source": entity.get(self.data_config.source_field, ""),
                    "source_name": entity.get(self.data_config.source_name_field, ""),
                    "lt_doc_id": entity.get(self.data_config.lt_doc_id_field, ""),
                    "chunk_id": entity.get(self.data_config.chunk_id_field, ""),
                    "hash_id": entity.get(self.data_config.hash_id_field, ""),
                }
            ))

        return results
