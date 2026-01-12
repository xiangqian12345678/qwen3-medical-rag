"""核心知识库模块"""
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker

from config.models import AppConfig, SearchRequest, SingleSearchRequest
from utils import create_embedding_client
from embed import Vocabulary, BM25Vectorizer, BM25SparseEmbedding

logger = logging.getLogger(__name__)


class KnowledgeBase:

    def __init__(self, app_config: AppConfig):
        """
        初始化知识库

        Args:
            app_config: 应用配置
        """
        self.app_config = app_config
        self.milvus_config = app_config.milvus
        self.dense_fields = app_config.dense_fields
        self.sparse_fields = app_config.sparse_fields
        self.data_config = app_config.data

        # 创建Milvus客户端
        self.client = MilvusClient(
            uri=self.milvus_config.uri,
            token=self.milvus_config.token
        )

        # 嵌入器字典
        self.embedders = {}

        # 初始化稠密向量嵌入器
        for field_name, field_config in self.dense_fields.items():
            embedder = create_embedding_client(field_config)
            self.embedders[field_config.index_field] = embedder

        # 初始化稀疏向量（BM25）
        for field_name, field_config in self.sparse_fields.items():
            if field_config.provider == "self":
                self._vocab = Vocabulary.load(field_config.vocab_path_or_name)
                if self._vocab is None:
                    raise FileNotFoundError(f"词表加载失败: {field_config.vocab_path_or_name}")

                self._bm25 = BM25Vectorizer(
                    vocab=self._vocab,
                    domain_model=field_config.domain_model,
                    k1=field_config.k1,
                    b=field_config.b
                )

                self.embedders[field_config.index_field] = BM25SparseEmbedding(self._vocab, self._bm25)

        logger.info(f"知识库初始化完成，集合: {self.milvus_config.collection_name}")

    def _encode_query(self, query: str, anns_field: str) -> Any:
        """
        编码查询

        Args:
            query: 查询文本
            anns_field: 检索字段

        Returns:
            编码后的查询向量/文本
        """
        if anns_field in self.embedders:
            return self.embedders[anns_field].embed_query(query)
        else:
            raise ValueError(f"未找到嵌入器: {anns_field}")

    def _build_ann_search_request(
            self,
            query: str,
            single_request: SingleSearchRequest
    ) -> AnnSearchRequest:
        """
        构建AnnSearchRequest

        Args:
            query: 查询文本
            single_request: 单个检索请求

        Returns:
            AnnSearchRequest对象
        """
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

    def _search(
            self,
            query: str,
            single_request: SingleSearchRequest,
            output_fields: List[str]
    ) -> List[Dict]:
        """
        单路检索

        Args:
            query: 查询文本
            single_request: 单个检索请求
            output_fields: 输出字段

        Returns:
            检索结果
        """
        data = self._encode_query(query, single_request.anns_field)

        result = self.client.search(
            collection_name=self.milvus_config.collection_name,
            data=[data],
            filter=single_request.expr,
            limit=single_request.limit,
            output_fields=output_fields,
            search_params={
                "metric_type": single_request.metric_type,
                "params": single_request.search_params
            },
            anns_field=single_request.anns_field
        )

        return result[0] if result else []

    def _hybrid_search(
            self,
            req: SearchRequest
    ) -> List[Dict]:
        """
        混合检索

        Args:
            req: 检索请求

        Returns:
            检索结果
        """
        # 构建子请求
        anns = []
        for single_req in req.requests:
            anns.append(self._build_ann_search_request(req.query, single_req))

        # 创建融合器
        if req.fuse.method == "rrf":
            ranker = RRFRanker(k=req.fuse.k)
        else:
            # weighted模式：从字典中提取权重值并转换为列表
            if req.fuse.weights:
                if isinstance(req.fuse.weights, dict):
                    weight_list = list(req.fuse.weights.values())
                else:
                    weight_list = req.fuse.weights
            else:
                # 默认权重：4个检索字段的平均权重
                weight_list = [0.25, 0.25, 0.25, 0.25]
            ranker = WeightedRanker(*weight_list)

        # 执行混合检索
        result = self.client.hybrid_search(
            collection_name=req.collection_name,
            reqs=anns,
            ranker=ranker,
            limit=req.top_k,
            output_fields=req.output_fields
        )

        return result[0] if result else []

    def search(self, req: SearchRequest) -> List[Document]:
        """
        执行知识库检索

        Args:
            req: 检索请求

        Returns:
            Document列表
        """
        if len(req.requests) == 1 and req.fuse is None:
            # 单路检索
            outputs = self._search(req.query, req.requests[0], req.output_fields)
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

            # 使用chunk字段作为主要内容
            results.append(Document(
                page_content=entity.get(self.data_config.chunk_field, ""),
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
                    "chunk_id": entity.get(self.data_config.chunk_id_field, -1),
                    "hash_id": entity.get(self.data_config.hash_id_field, ""),
                }
            ))

        return results
