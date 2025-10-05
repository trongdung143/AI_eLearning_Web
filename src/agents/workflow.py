from src.agents.qa.qa import QaAgent
from src.agents.writer.writer import  WriterAgent
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from src.agents.state import State
from src.agents.lecturer.lecturer import LecturerAgent
from langgraph.checkpoint.memory import MemorySaver

VALID_AGENTS = ["qa", "lecturer", "assessment"]


def route(state: State) -> str:
    type_request = state.get("type_request")
    if type_request in VALID_AGENTS:
        return type_request
    return "writer"

def start(state: State) -> State:
    return state

qa= QaAgent()
lecturer = LecturerAgent()
writer = WriterAgent()

workflow = StateGraph(State)


workflow.add_node("start", start)
workflow.add_node("qa", qa.process)
workflow.add_node("lecturer", lecturer.process)
workflow.add_node("writer", writer.process)
workflow.set_entry_point("start")

workflow.add_conditional_edges(
    "start",
    route,
    {
        "qa": "qa",
        "lecturer": "lecturer",
        # "assessment": "assessment",
    }
)
#
# for agent in VALID_AGENTS:
#     workflow.add_edge(agent, "writer")
workflow.add_edge("qa", "writer")
workflow.add_edge("lecturer", "writer")
workflow.set_finish_point("writer")

graph = workflow.compile(checkpointer=MemorySaver())

