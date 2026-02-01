# 性能监控优化说明

## 优化目标
为所有大模型交互添加耗时计算，解决 `generate_time` 显示为 `0.0` 的问题。

## 问题原因
原代码依赖 `msg.response_metadata.get("total_duration", 0)` 获取生成时间，但某些 LLM 提供商（如 Ollama）不会在响应元数据中返回 `total_duration` 字段，导致始终返回默认值 `0`。

## 解决方案

### 1. 核心工具函数 (`enhance/utils.py`)
- **`invoke_with_timing()`**: 统一的带计时和性能日志的 LLM 调用包装函数
  - 自动记录调用开始和结束时间
  - 计算生成耗时
  - 调用 `strip_think_get_tokens` 处理结果
  - 自动记录性能日志

### 2. 修改的文件

#### `enhance/query_enhance.py`
- 导入 `invoke_with_timing` 替代手动计时
- 所有 LLM 调用统一使用 `invoke_with_timing`
- 支持的阶段：`rewrite_query`, `ask`, `extract`

#### `enhance/recall_enhance.py`
- 导入 `invoke_with_timing` 替代手动计时
- 所有 LLM 调用统一使用 `invoke_with_timing`
- 支持的阶段：`multi_query`, `sub_query`, `superordinate_query`, `hypothetical_answer`

#### `answer/answer.py`
- 手动添加 `time.time()` 计时
- 记录性能信息到日志
- 输出包含 `msg`, `msg_len`, `msg_token_len`, `generate_time`

#### `enhance/filter_enhance.py`
- 为 `filter_low_correction_content` 添加计时
- 为 `filter_low_correction_doc_llm` 添加计时
- 记录生成时间和处理后的文档数量

### 3. 性能日志格式
所有 LLM 调用都会输出以下格式的日志：

```
INFO:root:  <stage_name>: {'msg': '...', 'msg_len': 100, 'msg_token_len': 50, 'generate_time': 1.23}
```

其中：
- `stage_name`: 阶段名称（如 `ask`, `extract`, `rewrite_query`, `answer` 等）
- `msg`: LLM 返回的消息内容（已去除思考标签）
- `msg_len`: 消息字符长度
- `msg_token_len`: 输出 token 数量
- `generate_time`: 生成时间（秒）

## 使用示例

### 基本用法
```python
from .utils import invoke_with_timing

ai = invoke_with_timing(
    (prompt | llm),
    {
        "background_info": state["background_info"],
        "question": state["curr_input"],
    },
    stage_name="ask"
)
```

### 手动计时
```python
import time

start_time = time.time()
ai_msg = (prompt | llm).invoke(inputs)
generate_time = time.time() - start_time

result = strip_think_get_tokens(ai_msg, generate_time=generate_time)
logger.info(f"  stage_name: {result}")
```

## 性能监控覆盖范围

| 阶段 | 文件 | 说明 |
|------|------|------|
| `ask` | `query_enhance.py` | 判断是否需要追问 |
| `extract` | `query_enhance.py` | 生成对话摘要 |
| `rewrite_query` | `query_enhance.py` | 查询改写 |
| `multi_query` | `recall_enhance.py` | 生成多个查询 |
| `sub_query` | `recall_enhance.py` | 拓展为子查询 |
| `superordinate_query` | `recall_enhance.py` | 生成上位查询 |
| `hypothetical_answer` | `recall_enhance.py` | 生成假设性回答 |
| `answer` | `answer/answer.py` | 生成最终答案 |
| `filter_low_correction_content` | `filter_enhance.py` | LLM 压缩过滤 |
| `filter_low_correction_doc_llm` | `filter_enhance.py` | LLM 文档过滤 |

## 注意事项
1. 所有耗时单位为秒（保留 3 位小数）
2. 如果 LLM 返回 `usage_metadata`，会记录 `output_tokens`
3. 日志通过 `logger.info()` 输出，可通过日志级别控制
4. `generate_time` 包含网络传输和模型推理时间
