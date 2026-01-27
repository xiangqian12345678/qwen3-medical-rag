"""数据库工厂类，用于创建和管理知识库实例"""
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Union

from ddgs import DDGS

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 知识库实例缓存
_ws_instance: Optional["WebSearcher"] = None


class WebSearcher:
    """知识库类，封装向量数据库检索功能"""

    def __init__(self, config: Union[Dict[str, Any], None] = None):
        self.config = config or {}

    def search(self, query: str, cnt: int = 5) -> list[Document]:
        """真正去拉取网页结果的搜索函数"""
        docs = []

        try:
            logger.info(f"开始使用 DuckDuckGo 搜索: {query}, 结果数: {cnt}")

            with DDGS() as ddgs:
                # 添加超时和重试机制
                results = list(ddgs.text(query, max_results=cnt))

                logger.info(f"DuckDuckGo 返回 {len(results)} 条原始结果")

                for i, hit in enumerate(results, start=1):
                    # 检查必需字段
                    if "body" not in hit or "title" not in hit:
                        logger.warning(f"结果 {i} 缺少必需字段,跳过")
                        continue

                    # 确保有内容
                    if not hit.get("body", "").strip():
                        logger.warning(f"结果 {i} 内容为空,跳过")
                        continue

                    id = hashlib.md5(hit["body"].encode('utf-8')).hexdigest()
                    docs.append(
                        Document(
                            page_content=hit["body"],
                            metadata={
                                "title": hit.get("title", "N/A"),
                                "url": hit.get("href", "N/A"),
                                "rank": i,
                                "source": "search",
                                "query": query,
                                "id": id
                            }
                        )
                    )

                    logger.info(f"成功解析 {len(docs)} 条文档")

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return docs


def get_ws(config: Dict[str, Any] = None) -> WebSearcher:
    """
    获取知识库实例（单例模式）

    Args:
        config: 配置字典

    Returns:
        知识库实例
    """
    global _ws_instance

    if _ws_instance is None:
        _ws_instance = WebSearcher(config)

    return _ws_instance


def reset_kb():
    """重置知识库实例（用于测试或重新初始化）"""
    global _ws_instance
    _ws_instance = None
