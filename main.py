# main.py

# --- 环境加载 ---
# 确保在导入任何其他模块之前首先加载环境变量
from dotenv import load_dotenv
load_dotenv()
# --- 环境加载结束 ---

from agent import app
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# 为每个对话创建一个唯一的ID，以便内存可以被隔离
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print("你好！我是一个高可靠性的对话代理。我们开始对话吧。（输入 '退出' 来结束）")

while True:
    user_input = input("你: ")
    if user_input.lower() == "退出":
        break
    
    # 使用 invoke 执行图，并获取最终状态
    final_state = app.invoke(
        {"messages": [HumanMessage(content=user_input)]}, config
    )
    
    # 从最终状态中提取并打印最后一条消息
    last_message = final_state.get("messages", [])[-1]
    if isinstance(last_message, AIMessage):
        last_message.pretty_print()
    elif final_state.get("execution_result"):
        print("\n--- 最终结果 ---")
        print(final_state["execution_result"])
        print("--------------------")
