from src.agents.rag.rag import RagAgent
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from src.agents.state import State
from langgraph.checkpoint.memory import MemorySaver

rag = RagAgent()

workflow = StateGraph(State)

workflow.add_node("rag", rag.process)
workflow.set_entry_point("rag")

graph = workflow.compile(checkpointer=MemorySaver())
input_state = {
    "messages": [HumanMessage(content="haha")],
    "thread_id": "conversation_id",
    "lesson_id": "",
    "task": "",
    "result": "",
    "file_path": "file_path",
}
