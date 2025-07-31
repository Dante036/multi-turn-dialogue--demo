# agent.py

import operator
from typing import TypedDict, Annotated, Any, List
from langchain_core.messages import BaseMessage, AIMessage

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    task_specification: dict
    execution_result: Any

# agent.py (继续)

from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel, Field

# 初始化一个不带任何工具的纯对话模型，供管理者使用
manager_llm = ChatTongyi(model="qwen-plus", temperature=0)

class TaskSpecification(BaseModel):
    """用于定义最终任务的结构化规范。"""
    query: str = Field(description="需要由工作者代理执行的、经过充分说明和澄清的用户查询。")
    requirements: List[str] = Field(description="从对话中提取的必须满足的约束或要求列表。")

class Clarification(BaseModel):
    """如果任务规范不完整，则用于向用户提问。"""
    question: str = Field(description="向用户提出的具体问题，以获取完成任务规范所需的缺失信息。")

class Clarification(BaseModel):
    """如果任务规范不完整，则用于向用户提问。"""
    question: str = Field(description="向用户提出的具体问题，以获取完成任务规范所需的缺失信息。")

class ManagerTools(BaseModel):
    """管理者可以使用的“工具”，实际上是决策路由。"""
    task_spec: TaskSpecification
    clarification: Clarification

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

# 为解析器创建一个独立的LLM实例
parser_llm = ChatTongyi(model="qwen-plus", temperature=0)

# --- 为手动解析创建提示模板 ---
parser_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个辅助工具，你的任务是根据用户和AI助手的对话历史，以及一个AI助手的最新回复，将该回复解析为指定的JSON格式。
            你必须选择两种格式之一：TaskSpecification（当任务已经完全清晰时）或Clarification（当还需要向用户提问时）。
            只输出JSON对象，不要包含任何其他文本或解释。"""
        ),
        (
            "human",
            """对话历史：
            {messages}
            
            AI助手的最新回复：
            {assistant_response}"""
        ),
    ]
)

# 将解析器LLM与Pydantic模型和提示绑定，创建解析链
parser = parser_prompt | parser_llm.with_structured_output(
    ManagerTools, include_raw=False
)

def manager_node(state: AgentState):
    """
    分析对话历史，决定是继续提问还是任务已清晰。
    """
    # 1. 首先，获取模型对于当前对话的原始、非结构化的回复
    raw_response = manager_llm.invoke(state['messages'])

    # 2. 然后，使用解析链来将原始回复强制转换为我们期望的结构
    # 我们将完整的消息历史和模型的原始回复都提供给解析器
    structured_response = parser.invoke(
        {
            "messages": state["messages"],
            "assistant_response": raw_response.content,
        }
    )

    # 3. 根据解析后的结构化输出来更新状态
    if isinstance(structured_response, TaskSpecification):
        return {"task_specification": structured_response.dict()}
    elif isinstance(structured_response, Clarification):
        return {"messages": [AIMessage(content=structured_response.question)]}
    
    # 如果解析失败，返回一个通用的澄清问题
    return {"messages": [AIMessage(content="抱歉，我没有完全理解。您能更详细地说明一下您的需求吗？")]}



# agent.py (继续)
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

# 1. 为工作者定义并初始化其专属工具
search_tool = TavilySearch(max_results=2)
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
        f"请根据以下详细信息执行任务：\n"
        f"查询: {task_spec['query']}\n"
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

# 4. 定义路由逻辑
def should_continue(state: AgentState):
    """
    路由逻辑的核心：决定下一步去哪里。
    """
    # 如果管理者刚刚提出了一个问题，那么流程应该暂停，等待用户输入。
    if isinstance(state['messages'][-1], AIMessage) and not state.get("task_specification"):
        return "END"
    # 如果任务规范已经形成，则进入工作者流程
    elif state.get("task_specification"):
        # 检查工作者是否需要调用工具
        if hasattr(state['messages'][-1], 'tool_calls') and state['messages'][-1].tool_calls:
            return "tools"
        # 如果工作者已经产出最终结果，则结束
        if state.get("execution_result"):
            return "END"
        # 否则，调用工作者
        return "worker"
    # 默认情况下，回到管理者继续澄清
    else:
        return "manager"

# 5. 添加条件边
workflow.add_conditional_edges(
    "manager",
    # 管理者节点后的路由逻辑：如果任务没定义好，就结束（等待用户输入）；否则交给工作者。
    lambda x: "worker" if x.get("task_specification") else "END",
    {"worker": "worker", "END": END}
)
workflow.add_conditional_edges(
    "worker",
    # 工作者节点后的路由逻辑：检查是否有工具调用
    lambda x: "tools" if hasattr(x['messages'][-1], 'tool_calls') and x['messages'][-1].tool_calls else END,
    {"tools": "tools", "END": END}
)
# 工具执行后，总是返回给工作者进行下一步评估
workflow.add_edge("tools", "worker")

# 6. 编译图
app = workflow.compile(checkpointer=memory)

