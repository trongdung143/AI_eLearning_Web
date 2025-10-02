from typing import Sequence
from time import time
import re, json
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools.base import BaseTool


from src.agents.base import BaseAgent
from src.agents.state import State
from langchain_core.messages import HumanMessage



class SupervisorAgent(BaseAgent):
    def __init__(self, agent_name, state, prompt, response_format, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name=agent_name,
            tools=tools,
            state=state,
            model=None,
        )

        self._prompt = prompt

        self._chain = self._prompt | self._model.with_structured_output(
            response_format
        )

    async def process(self, state: State) -> State:
        pass

