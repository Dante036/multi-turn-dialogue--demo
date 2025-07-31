# 实战指南：使用 LangGraph 构建高可靠性的对话代理

本指南将引导您完成构建一个高级对话代理的完整过程。该代理采用**职责分离的“管理者-工作者”架构**，利用 LangGraph 的循环和状态管理能力来克服传统多轮对话中的“对话迷失”问题。

### 核心架构思想

- **管理者 (Manager)**: 它的唯一职责是与用户进行对话，通过提问和澄清，将一个模糊的初始请求转化为一个**完整、无歧义的任务规范**。它**不持有**任何执行任务的工具（如网络搜索）。
- **工作者 (Worker)**: 它接收由管理者提供的清晰任务规范，并利用其持有的**一系列工具**（如网络搜索、代码生成等）来自主完成任务。

最终，您将拥有一个能够智能提问、澄清需求，并在完全理解用户意图后，才调用相应工具执行核心任务的健壮代理。

---

## 第一步：环境设置

首先，我们需要设置项目目录、Python 虚拟环境，并安装所有必需的库。

1.  **安装依赖库**
    我们将使用 Poetry 管理依赖。核心库包括 LangChain、LangGraph、Qwen (通义千问) 的模型以及专用于工作者的 Tavily 搜索工具。

    ```bash
    poetry add langchain langgraph langchain_community python-dotenv langchain-qianwen langchain-tavily
    ```

2.  **设置环境变量**
    在项目根目录下创建一个名为 `.env` 的文件，并填入您的 API 密钥。

    ```ini
    #.env 文件内容
    DASHSCOPE_API_KEY="sk-..."
    TAVILY_API_KEY="tvly-..."
    # (可选) 用于 LangSmith 调试
    LANGSMITH_API_KEY="ls__..."
    LANGCHAIN_TRACING_V2="true"
    ```

---

## 第二步：定义代理的“中央记忆体”（State）

LangGraph 的核心是 `State` 对象。我们的状态将保持不变，用于追踪整个对话和任务执行的流程。

在 `agent.py` 文件中添加以下代码：

```python
# agent.py

import operator
from typing import TypedDict, Annotated, Any, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    task_specification: dict
    execution_result: Any
```

---

## 第三步：构建图的节点（代理的逻辑单元）

我们将定义两个核心的逻辑单元：管理者和工作者。

### 1. 管理者节点 (Manager)

管理者的目标是**通过对话**形成清晰的任务规范。它不调用任何外部工具。

在 `agent.py` 文件中继续添加：

```python
# agent.py (继续)

from langchain_qianwen import ChatTongyi
from langchain_core.pydantic_v1 import BaseModel, Field

# 初始化一个不带任何工具的纯对话模型，供管理者使用
manager_llm = ChatTongyi(model="qwen-plus", temperature=0)

class TaskSpecification(BaseModel):
    """用于定义最终任务的结构化规范。"""
    query: str = Field(description="需要由工作者代理执行的、经过充分说明和澄清的用户查询。")
    requirements: List[str] = Field(description="从对话中提取的必须满足的约束或要求列表。")

class Clarification(BaseModel):
    """如果任务规范不完整，则用于向用户提问。"""
    question: str = Field(description="向用户提出的具体问题，以获取完成任务规范所需的缺失信息。")

class ManagerTools(BaseModel):
    """管理者可以使用的“工具”，实际上是决策路由。"""
    task_spec: TaskSpecification
    clarification: Clarification

def manager_node(state: AgentState):
    """
    分析对话历史，决定是继续提问还是任务已清晰。
    """
    # 将决策模型绑定到管理者的“工具”上
    manager_model = manager_llm.with_structured_output(ManagerTools)
    response = manager_model.invoke(state['messages'])
    
    if isinstance(response, TaskSpecification):
        # 任务已清晰，填充规范并准备移交给工作者
        return {"task_specification": response.dict()}
    elif isinstance(response, Clarification):
        # 任务不清晰，向用户提问
        return {"messages": [("ai", response.question)]}
    return {} # 默认无操作
```

### 2. 工作者节点 (Worker)

工作者接收明确的任务指示，并**使用其工具**来完成任务。

在 `agent.py` 文件中继续添加：

```python
# agent.py (继续)
from langchain_tavily import TavilySearchResults
from langgraph.prebuilt import ToolNode

# 1. 为工作者定义并初始化其专属工具
search_tool = TavilySearchResults(max_results=2)
worker_tools = [search_tool]

# 2. 为工作者创建一个绑定了工具的语言模型
worker_llm = ChatTongyi(model="qwen-plus", temperature=0)
worker_llm_with_tools = worker_llm.bind_tools(worker_tools)

# 3. 定义工作者节点
def worker_node(state: AgentState):
    """
    接收任务规范，并调用带有工具的模型来执行。
    """
    task_spec = state['task_specification']
    worker_prompt = (
        f"请根据以下详细信息执行任务：
"
        f"查询: {task_spec['query']}
"
        f"必须满足的要求: {', '.join(task_spec['requirements'])}"
    )
    
    # 工作者调用其带工具的模型
    result = worker_llm_with_tools.invoke(worker_prompt)
    
    # 如果模型决定调用工具，则将工具调用信息添加到消息历史中
    if result.tool_calls:
        return {"messages": [result]}
    
    # 如果模型直接返回结果，则任务完成
    return {"execution_result": result.content}

# 4. 定义一个标准的工具执行节点
# 这个节点将由工作者在需要时调用
tool_node = ToolNode(worker_tools)
```

---

## 第四步：连接节点（定义控制流）

现在我们用“边”来定义代理的控制流，体现“管理者 -> 工作者 -> 工具”的清晰路径。

在 `agent.py` 文件中继续添加：

```python
# agent.py (继续)

def router(state: AgentState):
    """
    根据当前状态决定下一个节点。
    """
    # 如果任务规范尚未形成，则保持在管理者循环中
    if not state.get("task_specification"):
        return "manager"
    
    # 如果任务规范已形成，则检查上一条消息
    last_message = state['messages'][-1]
    
    # 如果上一条消息是工具调用，则路由到工具节点
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # 如果任务已完成，则结束
    if state.get("execution_result"):
        return "END"
        
    # 否则，进入工作者节点执行任务
    return "worker"
```

---

## 第五步：组装并编译图

最后一步是将所有部分组装成一个可执行的图。

在 `agent.py` 文件中继续添加：

```python
# agent.py (继续)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. 初始化图和检查点
workflow = StateGraph(AgentState)
memory = MemorySaver()

# 2. 添加节点
workflow.add_node("manager", manager_node)
workflow.add_node("worker", worker_node)
workflow.add_node("tools", tool_node)

# 3. 设置入口点
workflow.set_entry_point("manager")

# 4. 添加条件边
workflow.add_conditional_edges(
    "manager",
    # 管理者节点后的路由逻辑很简单：如果任务没定义好，就回到自己；否则交给工作者。
    lambda x: "manager" if not x.get("task_specification") else "worker",
    {"manager": "manager", "worker": "worker"}
)
workflow.add_conditional_edges(
    "worker",
    # 工作者节点后的路由逻辑：检查是否有工具调用
    lambda x: "tools" if x['messages'][-1].tool_calls else END,
    {"tools": "tools", "END": END}
)
# 工具执行后，总是返回给工作者进行下一步评估
workflow.add_edge("tools", "worker")

# 5. 编译图
app = workflow.compile(checkpointer=memory)
```

---

## 第六步：运行和交互

交互代码 (`main.py`) 保持不变。

**如何执行：**

1.  确保您的 `.env` 文件已正确配置。
2.  在终端中运行 `main.py`：
    ```bash
    poetry run python main.py
    ```

**新架构下的交互示例：**

对话的前半部分将完全相同，代理通过提问来澄清需求。关键区别在于**工具调用发生在工作者阶段**。

```
你: 我需要写一个Python脚本。
================================ AI Message ================================

当然，我很乐意帮助。这个Python脚本应该做什么？它有什么具体的要求吗？
... (管理者持续提问，直到任务规范清晰) ...
你: 就用Tavily搜索“旧金山天气”吧，然后把温度和天气状况打印出来。

--- (此时，管理者完成任务，控制权交给工作者) ---
--- (工作者接收到清晰指令，决定调用搜索工具) ---

================================= Tool Call ==================================

tavily_search_results_json(query='旧金山天气')
================================== Tool Output =================================

... (Tavily的搜索结果) ...
================================ AI Message ================================


--- 最终结果 ---
好的，这是一个Python脚本... (工作者生成最终结果)
--------------------
```

在这个新架构中，管理者的循环是纯粹的对话，而工作者的循环则包含了“思考 -> 调用工具 -> 观察结果”的执行链，职责划分更加清晰。
