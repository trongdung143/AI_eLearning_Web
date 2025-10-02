from typing import Sequence
from time import time

from langchain_core.tools.base import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.agents.base import BaseAgent
from src.agents.writer.prompt import prompt
from src.agents.state import State
from langsmith import traceable


class WriterAgent(BaseAgent):
    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="writer",
            tools=tools,
            model=None,
        )

        self._prompt = prompt

        self._chain = self._prompt | self._model

    @traceable
    async def process(self, state: State) -> State:
        assigned_agents = state.get("assigned_agents")
        task = ""
        for agent in assigned_agents:
            task = task + f"{state.get("results").get(agent)[-1]}\n\n"
        result = None
        try:
            response = await self._chain.ainvoke(
                {
                    "task": [
                        HumanMessage(
                            content=f"{task}\n\n### Từ các kết quả từ các agent trước tổng hợp, diễn đạt lại cho người dùng"
                        )
                    ]
                }
            )

            result = f"### Kết quả (writer)\n{response.content}"
            current_tasks, current_results, _ = self.update_work(state, task, result)
            state.update(
                messages=[response],
                human=False,
                prev_agent=self._agent_name,
                next_agent=None,
                tasks=current_tasks,
                results=current_results,
            )
            print("writer")
        except Exception as e:
            print("ERROR ", self._agent_name)
        return state
