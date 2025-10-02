from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import StateGraph
from src.agents.analyst.analyst import AnalystAgent
from src.agents.rag.rag import RagAgent
from src.agents.writer.writer import WriterAgent
from src.agents.assigner.assigner import AssignerAgent
from src.agents.state import State
from src.agents.calculator.calculator import CalculatorAgent
from src.agents.coder.coder import CoderAgent
from src.agents.memory.memory import MemoryAgent
from src.agents.planner.planner import PlannerAgent
from src.agents.search.search import SearchAgent
from src.agents.emotive.emotive import emotiveAgent
from src.agents.simple.simple import SimpleAgent
app = StateGraph(State)

assigner = AssignerAgent()
analyst = AnalystAgent()
writer = WriterAgent()
calculator = CalculatorAgent()
coder = CoderAgent()
planner = PlannerAgent()
search = SearchAgent()
memory = MemoryAgent()
emotive = emotiveAgent()
rag = RagAgent()
simple = SimpleAgent()

VALID_AGENTS = [
        "calculator",
        "coder",
        "planner",
        "search",
        "emotive",
        "rag",
        "simple"
    ]

def route(state: State) -> str:
    assigned_agents = state.get("assigned_agents")
    next_agent = None
    for agent_name, tasks in assigned_agents.items():
        if tasks:
            next_agent = agent_name
            break
    if next_agent in VALID_AGENTS:
        return next_agent
    return "writer"

def noop(state: State) -> State:
    return state

app.add_node("assigner", assigner.process)
app.add_node("analyst", analyst.process)
app.add_node("writer", writer.process)
app.add_node("calculator", calculator.process)
app.add_node("coder", coder.process)
app.add_node("memory", memory.process)
app.add_node("planner", planner.process)
app.add_node("search", search.process)
app.add_node("emotive", emotive.process)
app.add_node("rag", rag.process)
app.add_node("simple", simple.process)
app.add_node("noop", noop)


app.set_entry_point("memory")
app.add_edge("memory", "analyst")
app.add_edge("analyst", "assigner")
app.add_edge("assigner", "noop")
app.add_conditional_edges(
    "noop",
    route,
    {
        "coder": "coder",
        "planner": "planner",
        "search": "search",
        "emotive": "emotive",
        "calculator": "calculator",
        "rag": "rag",
        "writer": "writer",
        "simple": "simple",
    },
)

for agent in VALID_AGENTS:
    app.add_edge(agent, "noop")
app.set_finish_point("writer")

graph = app.compile(checkpointer=MemorySaver())
