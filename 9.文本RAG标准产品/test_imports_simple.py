"""简单测试导入是否正常"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

print("Testing imports...")

try:
    print("1. Testing agent.milvus import...")
    from agent.milvus import llm_db_search, create_db_search_tool
    print("   ✓ agent.milvus import successful")
except Exception as e:
    print(f"   ✗ agent.milvus import failed: {e}")

try:
    print("2. Testing agent.search import...")
    from agent.search import llm_network_search, create_web_search_tool
    print("   ✓ agent.search import successful")
except Exception as e:
    print(f"   ✗ agent.search import failed: {e}")

try:
    print("3. Testing agent.kgraph import...")
    from agent.kgraph import llm_kgraph_search, create_kgraph_search_tool
    print("   ✓ agent.kgraph import successful")
except Exception as e:
    print(f"   ✗ agent.kgraph import failed: {e}")

try:
    print("4. Testing agent.rag_answer import...")
    from agent.rag_answer import rag_node, judge_node, finish_success, finish_fail
    print("   ✓ agent.rag_answer import successful")
except Exception as e:
    print(f"   ✗ agent.rag_answer import failed: {e}")

try:
    print("5. Testing agent.search_graph import...")
    from agent.search_graph import SearchGraph, SearchMessagesState
    print("   ✓ agent.search_graph import successful")
except Exception as e:
    print(f"   ✗ agent.search_graph import failed: {e}")

try:
    print("6. Testing agent.multi_dialogue_agent import...")
    from agent.multi_dialogue_agent import MultiDialogueAgent, MedicalAgentState
    print("   ✓ agent.multi_dialogue_agent import successful")
except Exception as e:
    print(f"   ✗ agent.multi_dialogue_agent import failed: {e}")

try:
    print("7. Testing agent.utils import...")
    from agent.utils import json_to_list_document, _should_call_tool, del_think
    print("   ✓ agent.utils import successful")
except Exception as e:
    print(f"   ✗ agent.utils import failed: {e}")

# AgentTools 已弃用并删除
# try:
#     print("8. Testing agent.tools import...")
#     from agent.tools import AgentTools
#     print("   ✓ agent.tools import successful")
# except Exception as e:
#     print(f"   ✗ agent.tools import failed: {e}")

print("\nAll import tests completed!")
