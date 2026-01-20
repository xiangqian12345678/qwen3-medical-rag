"""测试导入"""
import sys

print("开始测试导入...")

try:
    from agent import MultiDialogueAgent, MedicalAgentState
    print("✅ 成功导入 MultiDialogueAgent, MedicalAgentState")
except ImportError as e:
    print(f"❌ 导入 MultiDialogueAgent 失败: {e}")
    sys.exit(1)

try:
    from agent import SearchGraph, SearchMessagesState
    print("✅ 成功导入 SearchGraph, SearchMessagesState")
except ImportError as e:
    print(f"❌ 导入 SearchGraph 失败: {e}")
    sys.exit(1)

# AgentTools 已弃用并删除
# try:
#     from agent.tools import AgentTools
#     print("✅ 成功导入 AgentTools")
# except ImportError as e:
#     print(f"❌ 导入 AgentTools 失败: {e}")
#     sys.exit(1)

print("\n所有导入测试通过！")
