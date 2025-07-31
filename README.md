# 实战指南：使用 LangGraph 构建高可靠性的对话代理

本指南将引导您完成构建一个高级对话代理的完整过程。该代理采用“管理者-工作者”架构，利用 LangGraph 的循环和状态管理能力来克服传统多轮对话中的“对话迷失”问题。

最终，您将拥有一个能够智能提问、澄清需求、利用外部工具（如网络搜索），并在收集完所有必要信息后才执行核心任务的健壮代理。

---

## 第一步：环境设置

首先，我们需要设置项目目录、Python 虚拟环境，并安装所有必需的库。

1.  **安装依赖库**
    我们将使用 LangChain、LangGraph、OpenAI 的模型以及 Tavily 进行网络搜索。

    ```bash
    pip install -U langchain langgraph langchain-openai tavily-python python-dotenv
    ```

2.  **设置环境变量**
    在 `resilient_agent` 目录下创建一个名为 `.env` 的文件，并填入您的 API 密钥。这将使我们的代码能够安全地访问所需的服务。

    ```ini
    #.env 文件内容
    OPENAI_API_KEY="sk-..."
    TAVILY_API_KEY="tvly-..."
    # (可选，但强烈推荐) 用于调试和可观测性
    LANGSMITH_API_KEY="ls__..."
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_PROJECT="Resilient Agent Tutorial"
    ```

---

## 第二步：定义代理的“中央记忆体”（State）

LangGraph 的核心是 `State` 对象，它在整个工作流中被传递和修改。我们将定义一个全面的状态，以追踪对话的所有方面。

创建一个名为 `agent.py` 的文件，并添加以下代码：

```python
# agent.py

import operator
from typing import TypedDict, Annotated, Any, List
from langchain_core.messages import BaseMessage

# 使用 TypedDict 定义一个全面的状态模式
class AgentState(TypedDict):
    """
    代理状态的结构定义。

    Attributes:
        messages: 对话历史记录。
        task_specification: 从用户对话中提取的结构化任务参数。
        rephrased_query: 为 RAG 生成的独立查询。
        execution_result: 最终任务的执行结果。
    """
    # `add_messages` 是一个特殊的函数，它将新消息附加到列表中，而不是覆盖它
    messages: Annotated[list, operator.add]
    task_specification: dict
    rephrased_query: str
    execution_result: Any
```

---

## 第三步：为“管理者”代理配备工具

“管理者”代理的目标是收集信息。我们将为其提供一个强大的网络搜索工具（Tavily），以便它可以在需要时从外部获取信息来帮助澄清用户需求。

在 `agent.py` 文件中继续添加：

```python
# agent.py (继续)

from langchain_community.tools.tavily_search import TavilySearchResults

# 初始化工具
# 管理者代理将使用此工具来回答有关当前事件或它不知道的信息的问题。
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]
```

---

## 第四步：构建图的节点（代理的逻辑单元）

现在，我们将定义代理工作流中的每个计算步骤。每个节点都是一个 Python 函数，它接收当前状态，执行操作，并返回对状态的更新。

在 `agent.py` 文件中继续添加：

```python
# agent.py (继续)

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# 初始化语言模型
# 我们将使用一个支持工具调用的模型来驱动我们的代理
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 将工具绑定到模型，这样模型就知道它有哪些可用的工具
model_with_tools = model.bind_tools(tools)

# 1. 管理者节点：更新任务规范
# 这个节点是管理者的核心。它分析对话，并决定是需要澄清还是任务已准备好执行。

class TaskSpecification(BaseModel):
    """用于定义最终任务的结构化规范。"""
    query: str = Field(description="需要由工作者代理执行的、经过充分说明和澄清的用户查询。")
    requirements: List[str] = Field(description="从对话中提取的必须满足的约束或要求列表。")

class Clarification(BaseModel):
    """如果任务规范不完整，则用于向用户提问。"""
    question: str = Field(description="向用户提出的具体问题，以获取完成任务规范所需的缺失信息。")

class ManagerTools(BaseModel):
    """管理者可以使用的工具路由。"""
    task_spec: TaskSpecification
    clarification: Clarification

def manager_node(state: AgentState):
    """
    分析对话历史并决定下一步行动。
    - 如果任务清晰，则填充 task_specification。
    - 如果任务不清晰，则生成一个澄清问题。
    """
    # 使用绑定了澄清和任务规范工具的模型
    manager_model = model.with_structured_output(ManagerTools)
    
    response = manager_model.invoke(state['messages'])
    
    if isinstance(response, TaskSpecification):
        # 任务已准备好，填充规范并准备移交给工作者
        return {"task_specification": response.dict()}
    elif isinstance(response, Clarification):
        # 需要更多信息，向用户提问
        return {"messages": [("ai", response.question)]}
    else:
        # 备用逻辑
        return {"messages": [("ai", "无法确定下一步。")]}

# 2. 工作者节点：执行最终任务
# 这个节点只有在任务规范完全确定后才会被调用。

def worker_node(state: AgentState):
    """
    接收最终确定的任务规范并执行核心任务。
    """
    task_spec = state['task_specification']
    
    # 将规范格式化为一个清晰的提示
    worker_prompt = (
        f"请根据以下详细信息执行任务：
"
        f"查询: {task_spec['query']}
"
        f"必须满足的要求: {', '.join(task_spec['requirements'])}"
    )
    
    # 调用模型（可以绑定不同的工具）来执行任务
    result = model_with_tools.invoke(worker_prompt)
    
    return {"execution_result": result.content}

# 3. 工具执行节点
# 这是一个标准的 LangGraph 节点，用于执行模型请求的任何工具。
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

---

## 第五步：连接节点（定义控制流）

现在我们有了节点，需要用“边”将它们连接起来，告诉代理如何从一个步骤流向另一个步骤。我们将使用一个**条件边**来实现核心的循环逻辑。

在 `agent.py` 文件中继续添加：

```python
# agent.py (继续)

# 定义条件边
# 这个函数决定是从管理者循环到工具，还是在规范完成后移交给工作者。

def router(state: AgentState):
    """
    根据管理者节点的输出决定下一个节点。
    """
    if "task_specification" in state and state["task_specification"]:
        # 任务规范已填充，移交给工作者
        return "worker"
    else:
        # 任务规范不完整，继续管理者循环（可能调用工具或再次提问）
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            # 如果没有工具调用，并且规范不完整，则回到管理者进行下一次评估
            return "manager"

```

---

## 第六步：组装并编译图

最后一步是将我们定义的所有状态、节点和边组装成一个可执行的图。我们还将配置一个检查点（`checkpointer`）来自动保存对话历史，实现持久化记忆。

在 `agent.py` 文件中继续添加：

```python
# agent.py (继续)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. 初始化图和检查点
workflow = StateGraph(AgentState)
memory = MemorySaver()

# 2. 添加节点到图中
workflow.add_node("manager", manager_node)
workflow.add_node("tools", tool_node)
workflow.add_node("worker", worker_node)

# 3. 设置入口点
workflow.set_entry_point("manager")

# 4. 添加边
workflow.add_conditional_edges(
    "manager",
    router,
    {"worker": "worker", "tools": "tools", "manager": "manager"}
)
workflow.add_edge("tools", "manager") # 工具执行后，返回管理者进行评估
workflow.add_edge("worker", END) # 工作者执行完毕后，流程结束

# 5. 编译图
app = workflow.compile(checkpointer=memory)

# (可选) 可视化图
# 这将生成一个显示代理流程的图像文件。
try:
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("代理架构图已保存为 graph.png")
except Exception as e:
    print(f"无法生成图表: {e}")

```

---

## 第七步：运行和交互

现在，您的弹性对话代理已经准备就绪！以下是如何与它进行交互。创建一个名为 `main.py` 的新文件来运行代理。

```python
# main.py

from agent import app
import uuid

# 为每个对话创建一个唯一的ID，以便内存可以被隔离
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print("你好！我是一个高可靠性的对话代理。我们开始对话吧。（输入 '退出' 来结束）")

while True:
    user_input = input("你: ")
    if user_input.lower() == "退出":
        break
    
    # 将用户输入作为事件流式传输到应用中
    events = app.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
        elif "execution_result" in event and event["execution_result"]:
            print("
--- 最终结果 ---")
            print(event["execution_result"])
            print("--------------------")

```

**如何执行：**

1.  确保您的 `.env` 文件已正确配置。

2.  在终端中运行 `main.py`：

    ```bash
    python main.py
    ```

**交互示例：**

下面是一个演示代理如何工作的对话流程：

```
你好！我是一个高可靠性的对话代理。我们开始对话吧。（输入 '退出' 来结束）
你: 我需要写一个Python脚本。
================================ AI Message ================================
当然，我很乐意帮助。这个Python脚本应该做什么？它有什么具体的要求吗？
你: 它应该从一个网站上获取天气数据。
================================ AI Message ================================
好的，获取天气数据。您能告诉我应该从哪个具体的网站或API获取数据吗？另外，脚本需要处理哪些特定的城市？
你: 就用Tavily搜索“旧金山天气”吧，然后把温度和天气状况打印出来。
================================ AI Message ================================

================================= Tool Call ==================================
tavily_search_results_json(query='旧金山天气')
================================== Tool Output =================================
')]
================================ AI Message ================================

--- 最终结果 ---
好的，这是一个Python脚本，它使用Tavily API搜索旧金山当前的天气，然后打印出温度和天气状况。
... (此处为生成的代码)...
--------------------
你: 退出
```

在这个例子中，代理通过一系列澄清问题（“管理者”循环），将一个模糊的初始请求（“写一个Python脚本”）逐步细化为一个完整、可执行的任务规范，最后才调用“工作者”来生成最终的代码。
