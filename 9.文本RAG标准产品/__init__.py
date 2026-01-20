"""多轮对话Agent模块

该模块实现了多轮医疗对话Agent，具备以下功能：
1. 判断是否需要向用户追问关键信息（Ask）
2. 从多轮追问中抽取背景信息（Extract）
3. 判断是否需要拆分为多个子查询（Plan/Split）
4. 并行执行多个SearchGraph（Parallel Execute）
5. 汇总子查询结果，写回对话（Answer/Synthesize）
"""

__version__ = "1.0.0"
