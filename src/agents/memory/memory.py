from typing import Sequence

from langchain_core.messages import (
    SystemMessage,
    RemoveMessage,
)
from langchain_core.tools.base import BaseTool
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.agents.base import BaseAgent
from src.agents.memory.prompt import prompt
from src.agents.state import State


class MemoryAgent(BaseAgent):
    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="memory",
            tools=tools,
            model=None,
        )

        self._prompt = prompt

        self._chain = self._prompt | self._model

    async def process(self, state: State) -> State:
        task = None
        result = None
        messages = state.get("messages")
        print("length of messages: ", len(messages) + 1)
        try:
            if len(messages) >= 20:
                last_msg = messages[-1]
                task = "summary"
                response = await self._chain.ainvoke({"task": messages[:-1]})
                delete_msg = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
                messages = delete_msg + [SystemMessage(content=response.content), last_msg]
            else:
                task = "skiped"
                response = SystemMessage(content="messages < 20")
            result = response.content
            current_tasks, current_results, _ = self.update_work(state, task, result)
            state.update(
                messages=messages,
                next_agent="analyst",
                prev_agent=self._agent_name,
                tasks=current_tasks,
                results=current_results,
            )
            print("memory")
        except Exception as e:
            print("ERROR ", self._agent_name, f"\n{e}")
        return state
