from typing import Sequence
from langchain_core.tools.base import BaseTool
from time import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools.base import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.agents.base import BaseAgent
from src.agents.state import State
from src.agents.search.prompt import prompt


class SearchAgent(BaseAgent):
    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="search",
            tools=tools,
            model=None,
        )

        self._prompt = prompt

        self._chain = self._prompt | self._model

    async def process(self, state: State) -> State:
        task = self.get_task(state)
        result = None
        try:
            response = await self._chain.ainvoke({"task": [HumanMessage(content=task)]})
            result = f"### Kết quả tìm kiếm (search)\n{response.content}"
            current_tasks, current_results, assigned_agents = self.update_work(state, task, result)
            state.update(
                human=False,
                prev_agent=self._agent_name,
                tasks=current_tasks,
                results=current_results,
                assigned_agents=assigned_agents,
            )
            print("search")
        except Exception as e:
            print("ERROR ", self._agent_name, f"\n{e}")
        return state
