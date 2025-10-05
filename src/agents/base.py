from typing import Sequence

from langchain_core.tools.base import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.state import State
from src.config.setup import GOOGLE_API_KEY


class BaseAgent:

    def __init__(
        self,
            agent_name: str,
            state: type[State] = State,
            tools: Sequence[BaseTool] | None = None,
    ) -> None:
        """

        :rtype: None
        """
        self._tools = list(tools or [])
        self._agent_name = agent_name

        self._model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            disable_streaming=False,
        ).bind_tools(self._tools)

        self._sub_graph = StateGraph(state)

    async def process(self, state: State) -> State:
        return state

    # def response_filter(self, content: str) -> tuple:
    #     lines = content.strip().splitlines()
    #     last_line = lines[-1].strip()
    #     direction = json.loads(last_line.lower())
    #     content = "".join(lines[:-1]).strip()
    #     return (content, direction)

    def _set_subgraph(self):
        pass

    def get_subgraph(self) -> CompiledStateGraph:
        # graph = StateGraph(State)
        # graph.add_node(self._agent_name, self.process)
        # graph.add_node("human_node", human_node)

        # if self._tools:
        #     graph.add_node("tools", ToolNode(self._tools))
        #     graph.add_conditional_edges(
        #         self._agent_name,
        #         tools_condition,
        #         {"tools": "tools", "__end__": "human_node"},
        #     )
        #     graph.add_edge("tools", self._agent_name)
        # else:
        #     graph.add_edge(self._agent_name, "human_node")

        # graph.set_entry_point(self._agent_name)
        # graph.set_finish_point("human_node")
        return self._sub_graph.compile(name=self._agent_name)
