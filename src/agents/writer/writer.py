from typing import Sequence

from langchain_core.tools.base import BaseTool
from langsmith import traceable

from src.agents.base import BaseAgent
from src.agents.state import State
from src.agents.writer.prompt import prompt


class WriterAgent(BaseAgent):
    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="writer",
            tools=tools,
        )

        self._prompt = prompt

        self._chain = self._prompt | self._model

    @traceable
    async def process(self, state: State) -> State:
        content = state.get("result")
        question = state.get("task")
        try:
            response = await self._chain.ainvoke({"question": question, "content": content})
            result = response.content
            state.update(result=result, messages=[response])
        except Exception as e:
            print("ERROR ", self._agent_name)
        return state
