from typing import Sequence
from time import time
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools.base import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.base import BaseAgent
from src.agents.state import State
from src.agents.assigner.prompt import prompt
import ast

class AssignerResponseFormat(BaseModel):
    content: str = Field(description="Đầu ra là ví dụ {'rag': ['công việc 1 của rag', 'công việc 2 của rag'], 'coder': ['công việc 1 của coder', ]}")


class AssignerAgent(BaseAgent):
    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="assigner",
            tools=tools,
            model=None,
        )

        self._prompt = prompt

        self._chain = self._prompt | self._model.with_structured_output(
            AssignerResponseFormat
        )

    async def process(self, state: State) -> State:
        task = state.get("results").get(state.get("prev_agent"))[-1]
        result = None
        try:
            response = await self._chain.ainvoke({"assignment": [HumanMessage(content=f"{task}\n\n### Phân công công việc")]})
            result = f"### Phân công (assigner)\n{response.content}"
            current_tasks, current_results, _ = self.update_work(state, task, result)
            assigned_agents = ast.literal_eval(response.content)

            state.update(
                prev_agent=self._agent_name,
                tasks=current_tasks,
                results=current_results,
                assigned_agents=assigned_agents,
            )
            print("assigner", assigned_agents)
        except Exception as e:
            print("ERROR ", self._agent_name, f"\n{e}")
        return state
