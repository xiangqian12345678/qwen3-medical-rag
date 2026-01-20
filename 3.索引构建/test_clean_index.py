"""测试索引清理功能"""
import logging
from run_clean_index import IndexCleaner

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_index_cleaner():
    """测试索引清理工具"""
    print("=" * 70)
    print("测试 Milvus 索引清理工具")
    print("=" * 70)

    try:
        # 初始化清理工具
        print("\n🔌 连接 Milvus...")
        cleaner = IndexCleaner("index.yaml")

        # 显示统计信息
        print("\n" + "=" * 70)
        print("当前索引状态")
        print("=" * 70)
        stats = cleaner.get_statistics()
        print(f"\n📊 Collection: {stats['collection_name']}")
        print(f"  存在: {'✅ 是' if stats['exists'] else '❌ 否'}")
        print(f"  已加载: {'✅ 是' if stats['loaded'] else '❌ 否'}")
        print(f"  实体数量: {stats['entity_count']}")

        # 列出所有 Collection
        all_collections = cleaner.list_collections()
        print(f"\n  所有 Collection ({len(all_collections)}个):")
        for coll in all_collections:
            print(f"    - {coll}")

        # 测试清空数据（需要确认）
        print("\n" + "=" * 70)
        print("测试清空数据功能")
        print("=" * 70)
        print("⚠️  注意：此操作需要手动确认")
        print("    如需测试，请直接运行: python run_clean_index.py")

        # 关闭连接
        cleaner.close()
        print("\n✅ 测试完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_index_cleaner()
