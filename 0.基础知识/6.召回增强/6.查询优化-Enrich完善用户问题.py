from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1.模型初始化
llm = ChatTongyi(model="qwen-max")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")


# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# 2.意图识别
# 2.1 示例业务模板
templates = {
    "订机票": ["起点", "终点", "时间", "座位等级", "座位偏好"],
    "订酒店": ["城市", "入住日期", "退房日期", "房型", "人数"],
}

# 2.2 意图识别提示模板
intent_prompt = PromptTemplate(
    input_variables=["user_input", "templates"],
    template="根据用户输入 '{user_input}'，选择最合适的业务模板。可用模板如下：{templates}。请返回模板名称。"
)

# 2.3 创建意图识别链
intent_chain = intent_prompt | llm

# 2.4 模拟用户输入
user_input = "我想订一张长沙去北京的机票"

# 2.5 识别意图
intent = intent_chain.invoke({"user_input": user_input, "templates": str(list(templates.keys()))}).content
print("意图：", intent)

# 3.优化访问模版
# 3.1 优化模版构建
selected_template = templates.get(intent)
print("模板：", selected_template)

# 补充信息提示模板
info_prompt = f"""
    请根据用户原始问题和模板，判断原始问题是否完善。
    如果问题缺乏需要的信息，请生成一个友好的请求，明确指出需要补充的信息。
    若问题完善后，返回包含所有信息的完整问题。

    ### 原始问题    
    {user_input}

    ### 模板
    {",".join(selected_template)}                                   
                                           
    ### 输出示例
    {{
        "isComplete": true,
        "content": "`完整问题`"
    }}
    {{
        "isComplete": false,
        "content": "`友好的引导到需要补充信息`"
    }}                                       
"""

print(f"info_prompt: \n {info_prompt}")

# 3.2 基于优化模版访问
# 历史记录
chat_history = ChatMessageHistory()
# 聊天模版
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个信息补充助手，任务是分析用户问题是否完整。"),
        ("placeholder", "{history}"),  # 历史记录的占位
        ("human", "{input}"),
    ]
)

# 补充信息链
info_chain = prompt | llm

# 自动处理历史记录，将记录注入输入并在每次调用后更新它
# 1. 系统会根据当前会话 ID 获取聊天历史（通过提供的 lambda 函数）
# 2. 将当前输入消息（从 input 键获取）和聊天历史（放入 history 键）组合成最终输入
# 3. 调用 info_chain 进行处理
# 4. 通常还会将新的交互记录自动保存到聊天历史中
with_message_history = RunnableWithMessageHistory(
    info_chain,  # 被包装的链式处理对象（通常是一个 LangChain 链或类似的可运行对象）
    lambda session_id: chat_history,  # 消息历史存储的获取函数： 以 session_id 为参数，返回对应的 chat_history 对象
    input_messages_key="input",  # 指定输入消息在输入字典中的键名
    history_messages_key="history",  # 指定历史消息在输入字典中的键名
)

# 判断问题是否完整，如果不完整，则生成追问请求 调用被包装的链（info_chain），并自动处理消息历史的注入和管理。
# 1. input 键：对应 RunnableWithMessageHistory 初始化时设置的 input_messages_key="input"，表示当前用户消息的字段名
# 2. info_prompt：用户的实际输入内容（通常是一个问题或指令字符串）
# 3. session_id：标识对话会话的唯一键，用于从历史存储中获取或更新对话历史，在多用户场景中，通过不同 session_id 隔离各自的对话历史
info_request = with_message_history.invoke(input={"input": info_prompt},
                                           config={"configurable": {"session_id": "unused"}}).content

# 3.3 解析访问结果
parser = JsonOutputParser()
json_data = parser.parse(info_request)
# 循环判断是否完整，并提交用户补充信息
while json_data['isComplete'] is False:
    # 根据大模型的引导，用户补充信息
    user_answer = input(json_data['content'])
    # 提交用户补充信息，并判断问题是否完整
    info_request = with_message_history.invoke(input={"input": user_answer},
                                               config={"configurable": {"session_id": "unused"}}).content

    # 打印完整历史记录 确认是否有存储
    print("=" * 100)
    print("当前对话历史：")
    for message in chat_history.messages:
        print(f"{message.type}: {message.content}")
    print("=" * 100)

    try:
        json_data = parser.parse(info_request)
    except Exception as e:
        print("json parse error")
        break
print(info_request)
