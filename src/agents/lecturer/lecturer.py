from typing import Sequence

from langchain_core.tools.base import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document

from src.agents.base import BaseAgent
from src.agents.state import State
from src.agents.lecturer.prompt import prompt_lecturer

class LecturerState(dict):
    question: str
    genarate: str
    next_node: str
    document_path: str
    documents: list[Document]
    sermon: list[str]
    url_image: list[str]

class LecturerAgent(BaseAgent):
    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="lecturer",
            state=LecturerState,
            tools=tools,
        )

        self._prompt = prompt_lecturer

        self._chain = self._prompt | self._model
        self._set_subgraph()

    def _set_subgraph(self):
        pass

    def _read_document(self, ):
        pass

    async def process(self, state: State, config: RunnableConfig) -> State:
        return state
