"""测试网络搜索修复"""
import logging
from recall.search.web_searcher import get_ws, reset_kb

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 60)
print("测试网络搜索修复")
print("=" * 60)

# 重置单例
reset_kb()

# 获取 WebSearcher 实例
ws = get_ws()

# 测试搜索
test_query = "人工智能在医疗领域的应用"
print(f"\n搜索查询: {test_query}")
print("-" * 60)

try:
    results = ws.search(test_query, cnt=5)
    print(f"\n✓ 搜索成功!")
    print(f"  - 检索到 {len(results)} 条结果")

    if results:
        print(f"\n前 3 条结果:")
        for i, doc in enumerate(results[:3], 1):
            print(f"\n--- 结果 {i} ---")
            print(f"  标题: {doc.metadata.get('title', 'N/A')}")
            print(f"  URL: {doc.metadata.get('url', 'N/A')}")
            print(f"  内容: {doc.page_content[:200]}...")
    else:
        print("  ⚠ 未检索到任何结果")
        print("  可能的原因:")
        print("  1. 网络连接问题")
        print("  2. 搜索引擎暂时不可用")
        print("  3. API 限制")

except Exception as e:
    print(f"\n✗ 搜索失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
