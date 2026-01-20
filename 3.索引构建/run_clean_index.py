"""
Milvus 索引清理工具
提供安全的索引清理功能，包括删除 Collection、清空数据等
"""
import logging
from typing import Optional, List
from pymilvus import MilvusClient

from config import ConfigLoader
from collection import CollectionManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexCleaner:
    """Milvus 索引清理类"""

    def __init__(self, config_path: str = "index.yaml"):
        """
        初始化清理工具

        Args:
            config_path: 配置文件路径

        【输入示例】
        cleaner = IndexCleaner("index.yaml")

        【输出示例】
        None (清理工具已初始化)
        """
        # 加载配置
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config

        # 创建 Milvus 客户端
        self.milvus_config = self.config.milvus
        self.client = MilvusClient(
            uri=self.milvus_config.uri,
            token=self.milvus_config.token
        )
        self.collection_manager = CollectionManager(self.config)

        logger.info(f"✅ 索引清理工具已初始化")
        logger.info(f"   Milvus URI: {self.milvus_config.uri}")
        logger.info(f"   Collection: {self.milvus_config.collection_name}")

    def get_statistics(self, collection_name: Optional[str] = None) -> dict:
        """
        获取 Collection 统计信息

        Args:
            collection_name: Collection 名称，默认使用配置中的名称

        Returns:
            dict: 统计信息字典

        【输出示例】
        {
            "collection_name": "medical_knowledge",
            "entity_count": 12580,
            "exists": True,
            "loaded": True
        }
        """
        name = collection_name or self.milvus_config.collection_name

        # 检查 Collection 是否存在
        exists = self.client.has_collection(collection_name=name)

        if not exists:
            return {
                "collection_name": name,
                "entity_count": 0,
                "exists": False,
                "loaded": False
            }

        # 获取实体数量
        stats = {
            "collection_name": name,
            "entity_count": self.client.query(
                collection_name=name,
                filter="",
                output_fields=["pk"],
                limit=16384
            ).get("total", 0) if hasattr(self.client.query(
                collection_name=name,
                filter="",
                output_fields=["pk"],
                limit=1
            ), "total") else 0,
            "exists": True,
            "loaded": False
        }

        # 检查 Collection 是否已加载
        try:
            load_state = self.client.get_load_state(collection_name=name)
            stats["loaded"] = load_state.get("state") == "Loaded"
        except Exception:
            stats["loaded"] = False

        return stats

    def drop_collection(self, collection_name: Optional[str] = None, confirm: bool = False) -> bool:
        """
        删除指定的 Collection（索引和数据）

        Args:
            collection_name: Collection 名称，默认使用配置中的名称
            confirm: 是否确认删除，必须为 True 才能执行

        Returns:
            bool: 操作是否成功

        【输入示例】
        success = cleaner.drop_collection(confirm=True)

        【输出示例】
        即将删除 Collection: medical_knowledge
        该操作将删除所有索引和数据！
        ✅ 成功删除 Collection: medical_knowledge
        返回: True

        ⚠️ 警告：此操作不可逆，请谨慎使用！
        """
        if not confirm:
            logger.warning("⚠️ 警告：此操作将删除 Collection，请设置 confirm=True 确认")
            return False

        name = collection_name or self.milvus_config.collection_name

        # 检查 Collection 是否存在
        if not self.client.has_collection(collection_name=name):
            logger.warning(f"⚠️ Collection '{name}' 不存在")
            return False

        # 获取统计信息
        stats = self.get_statistics(name)
        logger.info(f"\n即将删除 Collection: {name}")
        logger.info(f"  实体数量: {stats['entity_count']}")
        logger.warning("⚠️ 该操作将删除所有索引和数据！")

        # 卸载 Collection（如果已加载）
        try:
            self.client.release_collection(collection_name=name)
            logger.info(f"✅ 已卸载 Collection: {name}")
        except Exception as e:
            logger.debug(f"卸载 Collection 跳过（可能未加载）: {e}")

        # 删除 Collection
        try:
            self.client.drop_collection(collection_name=name)
            logger.info(f"✅ 成功删除 Collection: {name}")
            return True
        except Exception as e:
            logger.error(f"❌ 删除 Collection 失败: {e}")
            return False

    def clear_data(self, collection_name: Optional[str] = None, confirm: bool = False) -> bool:
        """
        清空 Collection 中的所有数据（保留索引结构）

        Args:
            collection_name: Collection 名称，默认使用配置中的名称
            confirm: 是否确认清空，必须为 True 才能执行

        Returns:
            bool: 操作是否成功

        【输入示例】
        success = cleaner.clear_data(confirm=True)

        【输出示例】
        即将清空 Collection: medical_knowledge 中的所有数据
        保留索引结构，仅删除数据
        ✅ 成功清空 Collection: medical_knowledge
        返回: True

        ⚠️ 注意：此操作会删除所有数据，但保留索引结构
        """
        if not confirm:
            logger.warning("⚠️ 警告：此操作将清空 Collection 数据，请设置 confirm=True 确认")
            return False

        name = collection_name or self.milvus_config.collection_name

        # 检查 Collection 是否存在
        if not self.client.has_collection(collection_name=name):
            logger.warning(f"⚠️ Collection '{name}' 不存在")
            return False

        # 获取统计信息
        stats = self.get_statistics(name)
        logger.info(f"\n即将清空 Collection: {name} 中的所有数据")
        logger.info(f"  当前实体数量: {stats['entity_count']}")
        logger.info("  保留索引结构，仅删除数据")

        # 删除所有数据（通过删除 Collection 后重建的方式）
        # 注意：Milvus 没有直接的"清空数据"命令，需要重建 Collection
        try:
            # 获取 Collection 的描述
            from pymilvus import Collection
            collection = Collection(name)

            # 删除 Collection
            self.client.drop_collection(collection_name=name)

            # 重建 Collection（保留 Schema）
            self.collection_manager.create_collection()
            self.collection_manager.build_index()

            logger.info(f"✅ 成功清空 Collection: {name}")
            logger.info("✅ 索引结构已重建")
            return True
        except Exception as e:
            logger.error(f"❌ 清空 Collection 失败: {e}")
            return False

    def drop_all_collections(self, confirm: bool = False) -> bool:
        """
        删除所有 Collection（危险操作）

        Args:
            confirm: 是否确认删除，必须为 True 才能执行

        Returns:
            bool: 操作是否成功

        【输入示例】
        success = cleaner.drop_all_collections(confirm=True)

        【输出示例】
        即将删除所有 Collection (3个)
        ⚠️ 该操作将删除所有 Collection 的数据！
        ✅ 成功删除 3 个 Collection
        返回: True

        ⚠️ 警告：此操作不可逆，将删除所有 Collection！
        """
        if not confirm:
            logger.warning("⚠️ 警告：此操作将删除所有 Collection，请设置 confirm=True 确认")
            return False

        # 获取所有 Collection
        collections = self.client.list_collections()

        if not collections:
            logger.info("📭 当前没有 Collection")
            return True

        logger.info(f"\n即将删除所有 Collection ({len(collections)}个)")
        logger.warning("⚠️ 该操作将删除所有 Collection 的数据！")

        deleted_count = 0
        for collection_name in collections:
            try:
                self.client.release_collection(collection_name=collection_name)
                self.client.drop_collection(collection_name=collection_name)
                logger.info(f"✅ 已删除: {collection_name}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"❌ 删除失败 {collection_name}: {e}")

        logger.info(f"✅ 成功删除 {deleted_count} 个 Collection")
        return deleted_count == len(collections)

    def list_collections(self) -> List[str]:
        """
        列出所有 Collection 名称

        Returns:
            List[str]: Collection 名称列表

        【输出示例】
        ["medical_knowledge", "test_collection", "temp_collection"]
        """
        return self.client.list_collections()

    def close(self):
        """关闭连接"""
        logger.info("👋 索引清理工具已关闭")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """主函数：交互式清理工具"""
    print("=" * 70)
    print("Milvus 索引清理工具")
    print("=" * 70)

    try:
        # 解析命令行参数
        import argparse
        parser = argparse.ArgumentParser(description="Milvus 索引清理工具")
        parser.add_argument(
            "--config",
            type=str,
            default="index.yaml",
            help="配置文件路径"
        )
        parser.add_argument(
            "--auto",
            type=str,
            choices=["drop", "clear", "all"],
            help="自动执行操作：drop=删除Collection, clear=清空数据, all=删除所有"
        )
        parser.add_argument(
            "--yes",
            action="store_true",
            help="跳过确认步骤（自动模式下使用）"
        )
        args = parser.parse_args()

        # 初始化清理工具
        print("\n🔌 连接 Milvus...")
        cleaner = IndexCleaner(args.config)
        print("✅ 连接成功")

        # 自动模式
        if args.auto:
            confirm = args.yes
            if args.auto == "drop":
                success = cleaner.drop_collection(confirm=confirm)
            elif args.auto == "clear":
                success = cleaner.clear_data(confirm=confirm)
            elif args.auto == "all":
                success = cleaner.drop_all_collections(confirm=confirm)

            if success:
                print("\n✅ 操作成功完成")
            else:
                print("\n❌ 操作失败")
            cleaner.close()
            return

        # 交互模式
        # 显示当前统计信息
        print("\n" + "=" * 70)
        print("当前 Milvus 状态")
        print("=" * 70)
        stats = cleaner.get_statistics()
        print(f"\n📊 统计信息:")
        print(f"  Collection 名称: {stats['collection_name']}")
        print(f"  是否存在: {'✅ 是' if stats['exists'] else '❌ 否'}")
        print(f"  是否已加载: {'✅ 是' if stats['loaded'] else '❌ 否'}")
        print(f"  实体数量: {stats['entity_count']}")

        # 列出所有 Collection
        all_collections = cleaner.list_collections()
        print(f"\n  所有 Collection ({len(all_collections)}个):")
        if all_collections:
            for coll in all_collections:
                marker = "⭐" if coll == stats['collection_name'] else "  "
                print(f"    {marker} {coll}")
        else:
            print("    (无)")

        # 显示操作菜单
        print("\n" + "=" * 70)
        print("可用操作")
        print("=" * 70)
        print("1. 删除当前 Collection（删除索引和数据）")
        print("2. 清空当前 Collection 数据（保留索引结构）")
        print("3. 删除所有 Collection（危险操作）")
        print("4. 列出所有 Collection")
        print("5. 仅显示统计信息")
        print("0. 退出")
        print("=" * 70)

        # 获取用户输入
        choice = input("\n请选择操作 (0-5): ").strip()

        if choice == "1":
            print("\n⚠️ 严重警告：此操作将删除 Collection！")
            print("将删除所有索引和数据，此操作不可逆！")
            confirm = input("确认删除？(输入 'yes' 确认): ").strip().lower()
            if confirm == "yes":
                cleaner.drop_collection(confirm=True)
            else:
                print("❌ 操作已取消")

        elif choice == "2":
            print("\n⚠️ 警告：此操作将清空 Collection 数据！")
            print("将删除所有数据，但保留索引结构")
            confirm = input("确认清空？(输入 'yes' 确认): ").strip().lower()
            if confirm == "yes":
                cleaner.clear_data(confirm=True)
            else:
                print("❌ 操作已取消")

        elif choice == "3":
            print("\n⚠️ 严重警告：此操作将删除所有 Collection！")
            print("将删除所有 Collection 的数据和索引，此操作不可逆！")
            confirm = input("确认删除？(输入 'yes' 确认): ").strip().lower()
            if confirm == "yes":
                cleaner.drop_all_collections(confirm=True)
            else:
                print("❌ 操作已取消")

        elif choice == "4":
            print("\n📋 所有 Collection:")
            collections = cleaner.list_collections()
            if collections:
                for i, coll in enumerate(collections, 1):
                    print(f"  {i}. {coll}")
            else:
                print("  (无 Collection)")

        elif choice == "5":
            print("\n✅ 仅显示统计信息，不执行任何操作")

        elif choice == "0":
            print("\n👋 退出")

        else:
            print("\n❌ 无效的选择")

        # 关闭连接
        cleaner.close()
        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
