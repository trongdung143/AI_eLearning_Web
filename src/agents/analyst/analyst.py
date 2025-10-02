from typing import Sequence

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools.base import BaseTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.agents.analyst.prompt import prompt, prompt_supervisor
from src.agents.base import BaseAgent
from src.agents.state import State
from src.agents.supervisor.supervisor import SupervisorAgent
from langsmith import traceable

ANALYSIS = "### Phân tích yêu cầu (analyst)"
TRY_ANALYSIS = "### Phân tích lại yêu cầu (analyst)"

class AnalystState(dict):
    messages: list[BaseMessage]
    task: str
    result: str
    feedback_supervisor: str
    feedback_user: str
    analysis: str
    next_node: str
    prev_node: str

class SupervisorResponseFormatForAnalyst(BaseModel):
    next_node: str = Field(description="'next_node' là 'human', 'llm', '__end__'")
    content: str = Field(description="Feedback cho agent biết điểm chưa hoàn thành tốt.")


class SupervisorForAnalyst(SupervisorAgent):
    def __init__(self):
        super().__init__(agent_name="supervisor", state=AnalystState, prompt=prompt_supervisor, response_format=SupervisorResponseFormatForAnalyst)

    @traceable
    async def process(self, state: AnalystState) -> AnalystState:
        try:
            task = state.get("task")
            analysis = state.get("analysis")
            response = await self._chain.ainvoke(
                {"supervision": [HumanMessage(content=f"### Yêu cầu (user)\n{task}\n\n{analysis}")]}
            )

            feedback_supervisor = f"### Feedback (supervisor)\n{response.content}"

            if response.next_node == "llm":
                state.update(
                    feedback_supervisor=feedback_supervisor,
                    prev_node="supervisor",
                    next_node="llm",
                )
            elif response.next_node == "human":
                state.update(
                    result=analysis,
                    next_node="human",
                )
            elif response.next_node == "__end__":
                state.update(
                    result=analysis,
                    next_node="__end__",
                )

            print("supervisor in analyst agent")
        except Exception as e:
            print("ERROR ", "supervisor in analyst agent\n", e)
        return state


# class AnalystResponseFormat(BaseModel):
#     content: str = Field(
#         description="Phân tích lại yêu cầu một cách rõ ràng, có cấu trúc."
#     )
#     human: bool = Field(description="True nếu cần con người tham gia, False nếu không")
#     next_node: str = Field(description="Nếu sau khi phân tích cần tính toán thì chọn 'calculator', ngược lại 'writer'")


class AnalystAgent(BaseAgent):
    VALID_NODES = [
        "llm",
        "human",
        "__end__"
    ]
    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="analyst",
            tools=tools,
            state=AnalystState,
            model=None,
        )

        self._prompt = prompt
        self._chain = self._prompt | self._model
        self._supervisor = SupervisorForAnalyst()
        self._set_subgraph()

    def _set_subgraph(self):
        self._sub_graph.add_node("llm", self._llm)
        self._sub_graph.add_node("human", self._human)
        self._sub_graph.add_node("supervisor", self._supervisor.process)

        self._sub_graph.set_entry_point("llm")
        self._sub_graph.add_edge("llm", "supervisor")
        self._sub_graph.add_conditional_edges(
            "supervisor",
            self._route,
            {"human": "human", "llm": "llm", "__end__": "__end__"}
        )
        self._sub_graph.add_edge("human", "llm")

    def _route(self, state: AnalystState) -> str:
        next_node = state.get("next_node").strip()
        if next_node in self.VALID_NODES:
            return next_node
        return "__end__"

    def _human(self, state: AnalystState) -> AnalystState:
        analysis = state.get("analysis")
        marker = [ANALYSIS, TRY_ANALYSIS]
        analysis_part = None
        for mark in marker:
            if mark in analysis:
                analysis_part = analysis.split(mark, 1)[-1].strip()
                break
        edit = interrupt({"AIMessage": analysis_part})
        result = f"### Feedback (user)\n{edit}"
        state.update(
            feedback_user=result,
            prev_node="human",
            next_node="llm",
        )
        print("human in analyst agent")
        return state

    async def _llm(self, state: AnalystState) -> AnalystState:
        try:
            result = None
            analysis = None
            if state.get("prev_node") not in ["supervisor", "human"]:
                messages = state.get("messages")
                task = state.get("task")
                response = await self._chain.ainvoke({"task": messages[:-1] + [HumanMessage(content=f"### Yêu cầu (user)\n{task}")]})
                analysis = f"{ANALYSIS}\n{response.content}"


            elif state.get("prev_node") in ["supervisor", "human"]:
                feedback = None
                if state.get("prev_node") == "supervisor":
                    feedback = state.get("feedback_supervisor")
                else:
                    feedback = state.get("feedback_user")
                analysis = state.get("analysis")
                response = await self._chain.ainvoke({"task": [HumanMessage(content=f"{analysis}\n\n{feedback}\n\n### Từ feedback hãy sửa lại phân tích.")]})
                analysis = f"{TRY_ANALYSIS}\n{response.content}"

            state.update(
                analysis=analysis,
                next_node="supervisor",
                prev_node="llm",
            )
            print("llm in analyst agent")
        except Exception as e:
            print("ERROR ", "llm in analyst agent\n", e)
        return state

    async def process(self, state: State) -> State:
        messages = state.get("messages")
        task = messages[-1].content
        input_state = {
            "messages": messages,
            "task": task,
            "result": None,
            "feedback_supervisor": None,
            "feedback_user": None,
            "analysis": None,
            "next_node": None,
            "prev_node": "other",
        }
        sub_graph = self.get_subgraph()
        response = await sub_graph.ainvoke(input=input_state)
        current_tasks, current_results, _ = self.update_work(state, task, response.get("result"))
        state.update(
            tasks=current_tasks,
            results=current_results,
            next_agent="assigner",
            prev_agent=self._agent_name,
        )
        return state

