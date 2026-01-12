import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
from ..config.models import *
from .utils import create_embedding_client, create_llm_client
import hashlib
from pymilvus import MilvusClient, DataType, Collection, connections, FunctionType, Function, AnnSearchRequest, \
    RRFRanker, WeightedRanker
from functools import lru_cache
from pathlib import Path
from ..config.models import AppConfig
from ..embed.sparse import Vocabulary, BM25Vectorizer
from .insert import insert_rows
from copy import deepcopy
from langchain_core.tools import StructuredTool
from ..embed.bm25 import BM25SparseEmbedding

get_resolve_path = lambda path, file=__file__: (Path(file).parent / Path(path)).resolve()

logger = logging.getLogger(__name__)

AllFields = [
    "pk", "text", "summary", "document", "source", "source_name",
    "lt_doc_id", "chunk_id", "summary_dense", "text_dense", "text_sparse"
]


class MedicalHybridKnowledgeBase:
    """医疗混合知识库 - 支持多向量字段检索"""

    def __init__(self, app_config: AppConfig):

        self.milvus_config = app_config.milvus
        self.embedding_config = app_config.embedding

        # 创建多个嵌入模型实例
        self.summary_embedding = self._create_summary_embedding()
        self.text_embedding = self._create_text_embedding()

        # 向量存储实例
        self.client = MilvusClient(uri=self.milvus_config.uri, token=self.milvus_config.token)
        self.EMBEDDERS = {
            "summary_dense": self.summary_embedding,
            "text_dense": self.text_embedding
        }

        if self.embedding_config.text_sparse.provider == "self":
            # 如果自己管理词表，则还要创建一个BM25 Embedding
            self._vocab = Vocabulary.load(self.embedding_config.text_sparse.vocab_path_or_name)
            self._bm25 = BM25Vectorizer(
                vocab=self._vocab,
                domain_model=self.embedding_config.text_sparse.domain_model,
                k1=self.embedding_config.text_sparse.k1,
                b=self.embedding_config.text_sparse.b
            )
            self.EMBEDDERS["text_sparse"] = BM25SparseEmbedding(self._vocab, self._bm25)

    def _create_summary_embedding(self) -> Embeddings:
        """创建问题嵌入模型（用于summary_dense字段）"""
        return create_embedding_client(self.embedding_config.summary_dense)

    def _create_text_embedding(self) -> Embeddings:
        """创建文本嵌入模型（用于text_dense字段）"""
        return create_embedding_client(self.embedding_config.text_dense)

    def _create_collection(self):
        """ 使用原生 Milvus 客户端创建Collection"""
        # 断言检查：确保摘要向量维度与文本向量维度一致
        # 注意：这里存在一个逻辑错误，应该比较 summary_dense 和 text_dense 的维度
        # 多向量单行存储时，两个嵌入模型的向量维度必须相同，否则无法在同一行中存储
        assert self.embedding_config.summary_dense.dimension == self.embedding_config.summary_dense.dimension, "多向量单行存储时，两个嵌入模型嵌入向量维度必须相同"

        # 获取向量维度，用于定义 FLOAT_VECTOR 字段的 dim 参数
        dim = self.embedding_config.summary_dense.dimension

        # 如果配置了删除旧集合，则重新创建集合
        if self.milvus_config.drop_old:
            # 检查集合是否已存在，如果存在则删除
            if self.client.has_collection(collection_name=self.milvus_config.collection_name):
                self.client.drop_collection(collection_name=self.milvus_config.collection_name)

            # 创建新的 schema（集合结构定义）
            schema = MilvusClient.create_schema(
                auto_id=self.milvus_config.auto_id,  # 是否自动生成主键 ID
                enable_dynamic_field=True,  # 启用动态字段，允许插入未在 schema 中定义的字段
            )

            # 根据配置选择主键字段的数据类型
            if self.milvus_config.auto_id:
                # 自动生成 ID 时，主键为 INT64 类型
                schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
            else:
                # 使用外部 ID 时，主键为 VARCHAR 类型，最大长度 65535
                schema.add_field(field_name="pk", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True
            )
            schema.add_field(
                field_name="summary",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="document",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="source_name",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="lt_doc_id",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="chunk_id",
                datatype=DataType.INT64,
                max_length=65535
            )
            schema.add_field(
                field_name="summary_dense",
                datatype=DataType.FLOAT_VECTOR,
                dim=dim
            )
            schema.add_field(
                field_name="text_dense",
                datatype=DataType.FLOAT_VECTOR,
                dim=dim
            )
            schema.add_field(
                field_name="text_sparse",
                datatype=DataType.SPARSE_FLOAT_VECTOR
            )
            if self.embedding_config.text_sparse.provider == "Milvus":
                bm25_fn = Function(
                    name="bm25_text_to_sparse",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["text_sparse"],
                )
                schema.add_function(bm25_fn)

            self.client.create_collection(collection_name=self.milvus_config.collection_name, schema=schema)

            return self.client
        else:
            return self.client  # 如果不删除老集合，那就直接返回，不要创建

    def build_index(self):
        """ 构建合适的索引，构建完成之后load """
        index_params = self.client.prepare_index_params()
        # 为 summary_dense 字段（摘要稠密向量）添加 HNSW 索引
        # HNSW（Hierarchical Navigable Small World）是一种高效的近似最近邻搜索算法
        # index_name: 索引名称，用于管理和引用该索引
        # metric_type: COSINE 表示使用余弦相似度作为距离度量
        # params: HNSW 算法的参数
        #   - M: 每个节点在图中的最大连接数（默认值 16，这里设为 32），较大的 M 可以提高召回率但会增加内存使用
        #   - efConstruction: 构建索引时的搜索范围（默认值 40，这里设为 200），较大的值可以提高索引质量但会增加构建时间
        index_params.add_index(
            field_name="summary_dense",
            index_type="HNSW",
            index_name="summary_dense_index",
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 200}
        )
        index_params.add_index(
            field_name="text_dense",
            index_type="HNSW",
            index_name="text_dense_index",
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 200}
        )
        if self.embedding_config.text_sparse.provider == "self":
            # 为 text_sparse 字段（稀疏向量）添加稀疏倒排索引
            # SPARSE_INVERTED_INDEX: 稀疏倒排索引，适用于稀疏向量检索（如 BM25 向量）
            # index_name: 索引名称，用于管理和引用该索引
            # metric_type: IP（Inner Product）表示使用内积作为距离度量，适合 BM25 等稀疏向量
            # params: 倒排索引算法参数
            #   - inverted_index_algo: DAAT_MAXSCORE 是文档归并算法，采用 Document-At-A-Time + MaxScore 策略
            #     DAAT: 文档级别归并，逐文档计算得分
            #     MaxScore: 跳过不可能进入 Top-K 的文档，提高查询效率
            index_params.add_index(
                field_name="text_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                index_name="text_sparse_index",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"}
            )
        else:
            # 如果使用 Milvus 内置的 BM25 函数，则在索引配置中指定 BM25 相关参数
            # metric_type: BM25 表示使用 Milvus 内置的 BM25 算法
            # params: BM25 算法参数
            #   - inverted_index_algo: 同上，使用 DAAT_MAXSCORE 提高查询效率
            #   - bm25_k1: BM25 的 k1 参数，控制词频饱和度（默认 1.5，值越大词频影响越大）
            #   - bm25_b: BM25 的 b 参数，控制文档长度归一化程度（默认 0.75，值越大长度惩罚越强）
            index_params.add_index(
                field_name="text_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": self.embedding_config.text_sparse.k1,
                    "bm25_b": self.embedding_config.text_sparse.b
                }
            )
        self.client.create_index(
            collection_name=self.milvus_config.collection_name,
            index_params=index_params
        )
        self.client.load_collection(self.milvus_config.collection_name)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档，自动处理多向量字段

        Args:
            documents: 待添加的文档列表，每个文档应包含 summary、text 等字段

        Returns:
            List[str]: 返回添加的行数（注：实际返回的是 int 类型）
        """
        # 用于存储需要分词处理的文档文本
        tokenizer_docs = []

        # 存储最终要插入到 Milvus 的行数据
        rows = []
        for doc in documents:
            # 提取摘要和正文内容
            summary = doc.metadata.get("summary", "")
            text = doc.page_content

            # 如果 metadata 中没有预先计算的 dense 向量，则实时计算
            # 注意：上线时应删除这些实时计算逻辑，改为批量预处理
            if len(doc.metadata.get("summary_dense", [])) == 0:
                doc.metadata["summary_dense"] = self.EMBEDDERS["summary_dense"].embed_documents([summary])[0]

            if len(doc.metadata.get("text_dense", [])) == 0:
                doc.metadata["text_dense"] = self.EMBEDDERS["text_dense"].embed_documents([text])[0]

            # 收集需要进行稀疏向量编码的文本字段（当前未使用）
            tokenizer_docs.append(text)

            # 深拷贝 metadata 并过滤，只保留 AllFields 中定义的字段
            doc_dict = deepcopy(doc.metadata)
            filtered = {k: v for k, v in doc_dict.items() if k in AllFields}

            # 如果不使用自动 ID，则使用 hash_id 作为主键
            # 这样可以避免插入重复数据，相同 hash_id 的数据会被覆盖
            if not self.milvus_config.auto_id:
                filtered["pk"] = doc.metadata.get("hash_id", "")

            # 设置文本内容
            filtered["text"] = text

            # 如果是自己管理词表（provider="self"），则需要手动构建稀疏向量
            # 使用 BM25Embedding 对 summary 进行稀疏编码
            if self.embedding_config.text_sparse.provider == "self":
                filtered["text_sparse"] = self.EMBEDDERS["text_sparse"].embed_documents([summary])[0]

            rows.append(filtered)

        insert_rows(
            client=self.client,
            collection_name=self.milvus_config.collection_name,
            rows=rows,
            show_progress=False  # 小批量不显示进度条
        )
        return len(rows)

    def _encode_query(self, query, anns_field):
        """编码查询数据，根据不同的向量字段类型进行相应的编码处理

        Args:
            query: 用户查询文本
            anns_field: 要检索的向量字段名，可选值：
                - summary_dense: 摘要稠密向量
                - text_dense: 文本稠密向量
                - text_sparse: 文本稀疏向量（BM25）

        Returns:
            编码后的查询数据：
            - summary_dense/text_dense: 返回浮点数列表的稠密向量
            - text_sparse (self): 返回稀疏向量字典
            - text_sparse (Milvus): 返回原始查询字符串（Milvus 内置 BM25 会自动处理）
        """
        if anns_field != "text_sparse":
            # 稠密向量字段：使用对应的嵌入模型编码查询文本
            encoded_query = self.EMBEDDERS[anns_field].embed_query(query)
        else:
            # 稀疏向量字段：根据提供方式处理
            if self.embedding_config.text_sparse.provider == "self":
                # 自己管理词表：使用 BM25SparseEmbedding 编码为稀疏向量
                encoded_query = self.EMBEDDERS[anns_field].embed_query(query)
            else:
                # Milvus 内置 BM25 函数：直接返回查询文本
                # Milvus 会在搜索时使用内置的 BM25 函数自动处理文本分词和稀疏化
                encoded_query = query
        return encoded_query

    def _search(
            self,
            query: str,  # 查询的问题
            single_search_request: SingleSearchRequest,
            collection_name: str,
            output_fields: list[str]
    ):
        """ Milvus 原生的查询单个问题 https://milvus.io/docs/zh/filtered-search.md """
        data = self._encode_query(query=query, anns_field=single_search_request.anns_field)

        result = self.client.search(
            collection_name=collection_name,
            data=[data],
            filter=single_search_request.expr,
            limit=single_search_request.limit,
            output_fields=output_fields,
            search_params={
                "metric_type": single_search_request.metric_type,
                "params": single_search_request.search_params
            },
            anns_field=single_search_request.anns_field
        )
        return result

    def _build_ann_search_request(
            self,
            query,
            single_search_request: SingleSearchRequest
    ) -> AnnSearchRequest:
        """
        构建 AnnSearchRequest 子请求对象，用于混合检索中的单个向量检索

        AnnSearchRequest 是 Milvus 混合检索 API 中的子请求对象，用于指定在单个向量字段上进行检索的参数。
        在混合检索场景中，可以创建多个 AnnSearchRequest 对象，分别在不同的向量字段上执行检索，
        然后使用 RRFRanker 或 WeightedRanker 对结果进行融合。

        Args:
            query: 用户查询文本
            single_search_request: 单次检索请求配置，包含以下字段：
                - anns_field: 要检索的向量字段名（summary_dense/text_dense/text_sparse）
                - metric_type: 距离度量类型（COSINE/IP/BM25）
                - search_params: 检索参数（如 HNSW 的 ef 参数）
                - limit: 返回的文档数量限制
                - expr: 过滤表达式（可选项）

        Returns:
            AnnSearchRequest: Milvus 混合检索的子请求对象，包含：
                - data: 编码后的查询向量/文本（列表格式，支持批量查询）
                - anns_field: 检索的向量字段名
                - param: 检索参数配置
                - limit: 返回文档数量
                - expr: 过滤表达式

        Example:
            # 创建两个子请求，分别在 summary_dense 和 text_sparse 字段上检索
            req1 = _build_ann_search_request(query, SingleSearchRequest(anns_field="summary_dense", ...))
            req2 = _build_ann_search_request(query, SingleSearchRequest(anns_field="text_sparse", ...))
            # 使用 RRF 算法融合结果
            res = client.hybrid_search([req1, req2], ranker=RRFRanker(), limit=top_k)
        """
        # 根据向量字段类型编码查询数据
        # - summary_dense/text_dense: 返回稠密向量列表 [float, float, ...]
        # - text_sparse (self): 返回稀疏向量字典
        # - text_sparse (Milvus): 返回原始查询字符串
        encoded_data = self._encode_query(query=query, anns_field=single_search_request.anns_field)

        # 构建子请求参数字典
        search_param = {
            "data": [encoded_data],  # 查询数据，使用列表格式以支持批量查询
            "anns_field": single_search_request.anns_field,  # 指定在哪个向量字段上检索
            "param": {
                "metric_type": single_search_request.metric_type,  # 距离度量方式
                "params": single_search_request.search_params  # 检索算法参数
            },
            "limit": single_search_request.limit,  # 单次检索返回的文档数量
            "expr": single_search_request.expr  # 布尔过滤表达式，用于过滤文档
        }

        # 创建并返回 AnnSearchRequest 对象
        return AnnSearchRequest(**search_param)

    def _hybrid_search(
            self,
            search: SearchRequest
    ):
        """执行 Milvus 混合检索，在多个向量字段上同时检索并融合结果

        混合检索（Hybrid Search）的核心思想是在不同的向量表示上分别检索，然后将结果融合。
        例如：
        - 在 summary_dense 字段上检索：匹配语义相似的摘要
        - 在 text_sparse 字段上检索：匹配关键词精确匹配的文本
        - 使用 RRFRanker 或 WeightedRanker 融合两个结果集

        RRF (Reciprocal Rank Fusion) 算法：
            RRF(d) = Σ(1 / (k + rank_i(d)))
            其中 d 是文档，rank_i(d) 是文档在第 i 个结果集中的排名，k 是平滑参数
            RRF 的优点是无需额外训练，简单有效，适合融合不同排名算法的结果

        WeightedRanker 算法：
            WeightedScore(d) = Σ(w_i * score_i(d))
            其中 w_i 是第 i 个结果集的权重，score_i(d) 是文档的原始分数
            适用于需要根据各检索器可信度分配权重的场景

        Args:
            search: 混合检索请求配置，包含以下字段：
                - query: 用户查询文本
                - collection_name: 集合名称
                - requests: 单次检索请求列表（SingleSearchRequest 列表）
                - fuse: 结果融合配置
                    * method: 融合方法，可选 "rrf" 或 "weighted"
                    * k: RRF 的平滑参数（仅当 method="rrf" 时使用）
                    * weights: 各结果集的权重列表（仅当 method="weighted" 时使用）
                - limit: 最终返回的文档数量
                - output_fields: 需要返回的字段列表

        Returns:
            List[List[Dict]]: 混合检索结果，嵌套结构：
                外层：每个查询对应一个结果列表（支持批量查询）
                内层：每条结果包含文档 ID、距离/分数、以及 output_fields 指定的字段

        Example:
            # 在 summary_dense 和 text_sparse 两个字段上检索，使用 RRF 融合
            search = SearchRequest(
                query="如何治疗高血压",
                collection_name="medical_kb",
                requests=[
                    SingleSearchRequest(anns_field="summary_dense", limit=50, ...),
                    SingleSearchRequest(anns_field="text_sparse", limit=50, ...)
                ],
                fuse=FuseConfig(method="rrf", k=60),
                limit=10,
                output_fields=["text", "summary", "source"]
            )
            results = _hybrid_search(search)
        """
        # 构建所有子检索请求
        # 每个 SingleSearchRequest 对应一个向量字段的检索
        anns = []
        for item in search.requests:
            anns.append(
                self._build_ann_search_request(
                    query=search.query,
                    single_search_request=item
                )
            )

        # 根据融合方法创建对应的 Ranker 对象
        if search.fuse.method == "rrf":
            # RRF (Reciprocal Rank Fusion): 倒数排名融合
            # k 是平滑参数，默认值 60，用于平衡不同排名位置的权重
            rank = RRFRanker(search.fuse.k)
        elif search.fuse.method == "weighted":
            # WeightedRanker: 加权融合
            # weights 是各检索结果的权重列表，需要与 search.requests 长度相同
            # 例如 [0.7, 0.3] 表示第一个检索结果权重 70%，第二个权重 30%
            rank = WeightedRanker(*search.fuse.weights)

        # 执行混合检索
        result = self.client.hybrid_search(
            collection_name=search.collection_name,
            reqs=anns,  # 子检索请求列表
            ranker=rank,  # 结果融合策略
            limit=search.limit,  # 最终返回的文档数量
            output_fields=search.output_fields  # 需要返回的字段
        )

        return result

    def search(self, req: SearchRequest) -> List[Document]:
        """执行知识库检索，支持单向量检索和混合检索

        根据请求中的检索策略（单字段或多字段）自动选择合适的检索方法：
        - 单字段检索：使用 _search 方法在单个向量字段上检索
        - 混合检索：使用 _hybrid_search 方法在多个向量字段上检索并融合结果

        检索结果会被封装为 LangChain Document 对象，便于后续的 RAG 流程使用。

        Args:
            req: 检索请求配置，包含以下字段：
                - query: 用户查询文本
                - collection_name: 集合名称
                - requests: 单次检索请求列表（SingleSearchRequest 列表）
                - fuse: 结果融合配置（混合检索时使用）
                - limit: 返回的文档数量
                - output_fields: 需要返回的字段列表

        Returns:
            List[Document]: 检索结果列表，每个 Document 包含：
                - page_content: 文档正文文本
                - metadata: 元数据字典，包含 pk、distance、chunk_id、summary、document、source、source_name、lt_doc_id

        Example:
            # 单字段检索示例
            req = SearchRequest(
                query="如何治疗高血压",
                collection_name="medical_kb",
                requests=[
                    SingleSearchRequest(anns_field="summary_dense", limit=10, ...)
                ],
                limit=10,
                output_fields=["text", "summary", "source"]
            )
            results = kb.search(req)

            # 混合检索示例
            req = SearchRequest(
                query="如何治疗高血压",
                collection_name="medical_kb",
                requests=[
                    SingleSearchRequest(anns_field="summary_dense", limit=50, ...),
                    SingleSearchRequest(anns_field="text_sparse", limit=50, ...)
                ],
                fuse=FuseConfig(method="rrf", k=60),
                limit=10,
                output_fields=["text", "summary", "source"]
            )
            results = kb.search(req)
        """
        if len(req.requests) == 1:
            # 只有一个请求搜索，走普通的search
            # 使用 _search 方法在单个向量字段上执行检索
            # 这种方式适用于只关注某个特定字段的场景，如只匹配摘要的语义相似度
            outputs = self._search(
                req.query,
                req.requests[0],
                req.collection_name,
                req.output_fields
            )[0]  # 批量中的第一条，这里先不支持批量查询
        else:
            # 有多个请求搜索，走混合search
            # 使用 _hybrid_search 方法在多个向量字段上检索并融合结果
            # 混合检索可以结合不同字段的检索优势，如语义匹配（dense）+ 关键词匹配（sparse）
            outputs = self._hybrid_search(req)[0]

        results = []

        for i in range(len(outputs)):  # 封装获得 List[Document]
            item = outputs[i]
            results.append(
                Document(
                    page_content=item.get("text", ""),  # 文档正文作为 page_content
                    metadata={
                        "pk": item.get("pk", ""),  # 主键 ID
                        "distance": item.get("distance", 99999),  # 相似度距离，越小越相似
                        "chunk_id": item.get("chunk_id", -1),  # 文档块 ID
                        "summary": item.get("summary", ""),  # 文档摘要
                        "document": item.get("document", ""),  # 完整文档
                        "source": item.get("source", ""),  # 数据来源 URL
                        "source_name": item.get("source_name", ""),  # 数据源名称
                        "lt_doc_id": item.get("lt_doc_id", ""),  # 长文档 ID
                    },
                )
            )

        return results
