"""演示大括号转义的必要性"""

# 场景1: 不转义 - 部分情况可工作
print("=" * 50)
print("场景1: 不转义大括号 (部分情况可工作)")
print("=" * 50)

format_instructions = '请按以下JSON格式输出: {"need_ask": true, "questions": ["问题1"]}'

# 这种情况下可以工作，因为 {format_instructions} 是合法的变量占位符
result = "系统提示: {format_instructions}".format(format_instructions=format_instructions)
print(f"结果: {result}")

print()
print("但是! 如果包含空大括号或嵌套结构,就会出错:")
print()

bad_template = "输出格式: {} 和 {{name}}"
try:
    result2 = "提示: {template}".format(template=bad_template)
    print(f"结果: {result2}")
except (KeyError, ValueError) as e:
    print(f"❌ 错误: {type(e).__name__}: {e}")

print()


# 场景2: 转义 - 正确示例
print("=" * 50)
print("场景2: 转义大括号 (正确)")
print("=" * 50)

format_instructions = '请按以下JSON格式输出: {"need_ask": true, "questions": ["问题1"]}'

# 先转义大括号
# 转义目的：防止大括号被 .format() 误认为是占位符
escaped_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
print(f"转义前: {format_instructions}")
print(f"转义后: {escaped_instructions}")

# 现在可以安全使用.format()
result = "系统提示: {format_instructions}".format(format_instructions=escaped_instructions)
print(f"结果: {result}")

print()


# 场景3: 实际项目中的用法对比
print("=" * 50)
print("场景3: 项目实际对比")
print("=" * 50)

# 模拟 Pydantic 输出的格式说明
mock_format_instructions = """
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"name": {"title": "Name", "type": "string"}}, "required": ["name"], "title": "User", "type": "object"}

Here is the output schema:
{"properties": {"need_ask": {"title": "Need Ask", "type": "boolean"}, "questions": {"items": {"type": "string"}, "title": "Questions", "type": "array"}}, "required": ["need_ask"], "title": "AskMess", "type": "object"}
"""

original_format = mock_format_instructions
print(f"原始格式说明 (前100字符): {original_format[:100]}...")
print()

# 不转义会出错 (因为格式说明中包含空大括号或复杂嵌套结构)
try:
    template = "格式要求: {format_instructions}".format(format_instructions=original_format)
    print("不转义成功 (不应该看到这行)")
except (KeyError, ValueError, IndexError) as e:
    print(f"❌ 不转义错误: {type(e).__name__}: {str(e)[:100]}...")

print()

# 转义后正常工作
escaped_format = original_format.replace("{", "{{").replace("}", "}}")
template = "格式要求: {format_instructions}".format(format_instructions=escaped_format)
print(f"✅ 转义成功,长度: {len(template)} 字符")
print(f"✅ 还原后包含: {{'need_ask': true, 'questions': ['问题'][:50]}}...")

print()
print("=" * 50)
print("总结")
print("=" * 50)
print("1. 简单的 {variable} 可以直接使用")
print("2. 但 Pydantic 的格式说明包含复杂的 JSON Schema,有各种嵌套大括号")
print("3. 为了安全起见,必须将所有 { 和 } 转义为 {{ 和 }}")
print("4. .format() 会自动将 {{ 和 }} 还原为 { 和 }")
