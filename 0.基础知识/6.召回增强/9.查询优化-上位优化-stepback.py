from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

"""
核心思想
    Step Back 是一种通过让模型先回答更抽象的“上位问题”（step-back question），再基于抽象答案解决原问题的技术。
    其灵感来源于人类思考复杂问题时，会先退一步思考更通用的原则。

关键步骤
1. 生成上位问题：
    从原始问题中提取一个更抽象、更本质的问题。
    示例：
    原问题 → "AlphaGo 如何击败李世石？"
    上位问题 → "强化学习在棋类游戏中的基本原理是什么？"

2. 回答上位问题：
    先获取抽象问题的答案（通用知识）。

3. 结合解决原问题：
    用抽象答案作为上下文，推导出原问题的具体答案。
"""
# 1.模型初始化
llm = ChatTongyi(model="qwen-max")

# 2.上位问题模版
step_back_prompt = PromptTemplate.from_template(
    """
    基于以下问题，生成一个更抽象的上位问题：
    原始问题: {original_question}
    上位问题:
    """
)

# 3. 生成上位问题
question = "AlphaGo 如何击败李世石？"
abstract_answer = llm.invoke(step_back_prompt.format(original_question=question))
print(f"上位问题：\n{abstract_answer}")

# 4.结合两者回答原问题
final_prompt = f"""请基于以下信息回答问题：
    上位问题: {abstract_answer}
    原始问题: {question}
    最终答案:"""
result = llm.invoke(final_prompt)
print('-' * 20 + '结合上位问题和原问题回答' + '-' * 20)
print(f"final_prompt:\n {final_prompt}")
print('答案：\n f{result}')
