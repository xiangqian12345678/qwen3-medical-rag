"""医疗知识库 - 主类"""
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from collection import CollectionManager
from config import SearchRequest, SingleSearchRequest, AnnsField, FusionSpec
from insert import insert_rows, upsert_rows
from searcher import KnowledgeBaseSearcher
from vectorizer import DocumentVectorizer

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    知识库 - 完整的索引构建和检索功能
    """

    def __init__(self, config_path: Optional[str] = None):
        """初始化知识库

        Args:
            config_path: 配置文件路径，默认使用 index.yaml
        """
        from config import ConfigLoader

        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config

        # 创建 Collection 管理器
        self.collection_manager = CollectionManager(self.config)
        self.client = self.collection_manager.client

        # 创建文档向量化器
        self.document_vectorizer = DocumentVectorizer(self.config)

        # 检索器（延迟初始化）
        self._searcher: Optional[KnowledgeBaseSearcher] = None

    # =============================================================================
    # Collection 管理
    # =============================================================================

    def create_collection(self):
        """创建 Collection"""
        return self.collection_manager.create_collection()

    def build_index(self):
        """构建索引"""
        return self.collection_manager.build_index()

    def drop_collection(self):
        """删除 Collection"""
        return self.collection_manager.drop_collection()

    def initialize(self):
        """初始化：创建 Collection 和索引"""
        self.create_collection()
        self.build_index()
        logger.info("知识库初始化完成")

    # =============================================================================
    # 文档添加
    # =============================================================================

    def add_documents(
            self,
            documents: List[Document],
            show_progress: bool = False
    ) -> int:
        """添加文档到知识库

        对于 list 类型字段（如 questions），每个元素会展开为单独的行。

        Args:
            documents: 文档列表
            show_progress: 是否显示进度条

        Returns:
            int: 插入的行数
        """
        rows = self.document_vectorizer.vectorize_documents_batch(documents)

        insert_rows(
            client=self.client,
            collection_name=self.config.milvus.collection_name,
            rows=rows,
            show_progress=show_progress
        )

        return len(rows)

    def upsert_documents(
            self,
            documents: List[Document],
            show_progress: bool = False
    ) -> int:
        """插入或更新文档

        Args:
            documents: 文档列表
            show_progress: 是否显示进度条

        Returns:
            int: 处理的行数
        """
        rows = self.document_vectorizer.vectorize_documents_batch(documents)

        upsert_rows(
            client=self.client,
            collection_name=self.config.milvus.collection_name,
            rows=rows,
            show_progress=show_progress
        )

        return len(rows)

    # =============================================================================
    # 检索
    # =============================================================================

    @property
    def searcher(self) -> KnowledgeBaseSearcher:
        """获取检索器（延迟初始化）"""
        if self._searcher is None:
            self._searcher = KnowledgeBaseSearcher(self.config, self.client)
        return self._searcher

    def search(
            self,
            query: str,
            anns_fields: Optional[List[AnnsField]] = None,
            limit: int = 5,
            top_k: int = 50,
            output_fields: Optional[List[str]] = None,
            fuse: Optional[bool] = True,
            deduplicate: bool = True
    ) -> List[Dict[str, Any]]:
        """执行检索

        Args:
            query: 查询文本
            anns_fields: 检索字段列表，默认使用所有配置的字段
            limit: 最终返回数量
            top_k: 每个检索返回的top_k数量
            output_fields: 输出字段列表
            fuse: 是否进行向量融合
            deduplicate: 是否按 origin_pk 去重

        Returns:
            List[Dict[str, Any]]: 检索结果
        """
        # 构建检索请求
        requests = []

        if anns_fields is None:
            # 使用所有启用的字段
            anns_fields = []
            for field_name, field_config in self.config.dense_fields.items():
                if field_config.embed:
                    anns_fields.append(field_config.index_field)
            for field_name, field_config in self.config.sparse_fields.items():
                if field_config.embed:
                    anns_fields.append(field_config.index_field)
        else:
            # 转换为 index_field 名称
            anns_fields_converted = []
            for field in anns_fields:
                # 如果传入的是字段名（如 questions），转换为 index_field（如 questions_dense）
                found = False
                for field_name, field_config in self.config.dense_fields.items():
                    if field == field_name or field == field_config.index_field:
                        anns_fields_converted.append(field_config.index_field)
                        found = True
                        break
                for field_name, field_config in self.config.sparse_fields.items():
                    if field == field_name or field == field_config.index_field:
                        anns_fields_converted.append(field_config.index_field)
                        found = True
                        break
                if not found:
                    logger.warning(f"未找到字段配置: {field}")
            anns_fields = anns_fields_converted

        for field in anns_fields:
            # 获取字段配置
            metric_type = "COSINE"
            search_params = {"ef": 64}

            for field_name, field_config in self.config.dense_fields.items():
                if field_config.index_field == field:
                    metric_type = field_config.metric_type
                    search_params = field_config.search_params
                    break
            for field_name, field_config in self.config.sparse_fields.items():
                if field_config.index_field == field:
                    metric_type = field_config.metric_type
                    search_params = {}
                    break

            requests.append(
                SingleSearchRequest(
                    anns_field=field,
                    metric_type=metric_type,
                    search_params=search_params,
                    limit=top_k
                )
            )

        # 输出字段
        if output_fields is None:
            output_fields = self.config.default_search.output_fields

        # 融合配置
        fusion_spec = None
        if fuse:
            fusion_config = self.config.fusion
            fusion_spec = FusionSpec(
                method=fusion_config.method,
                k=fusion_config.k,
                weights=fusion_config.weights
            )

        # 构建检索请求
        search_request = SearchRequest(
            query=query,
            collection_name=self.config.milvus.collection_name,
            requests=requests,
            output_fields=output_fields,
            fuse=fusion_spec,
            top_k=top_k,
            limit=limit
        )

        # 执行检索
        results = self.searcher.search(search_request)

        # 去重
        if deduplicate:
            results = self.searcher.deduplicate_by_origin_pk(results)

        return results

    def simple_search(
            self,
            query: str,
            anns_field: AnnsField,
            limit: int = 5
    ) -> List[Dict[str, Any]]:
        """单路检索

        Args:
            query: 查询文本
            anns_field: 检索字段
            limit: 返回数量

        Returns:
            List[Dict[str, Any]]: 检索结果
        """
        return self.search(
            query=query,
            anns_fields=[anns_field],
            limit=limit,
            fuse=False,
            deduplicate=False
        )
