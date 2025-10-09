from typing import Sequence
import logging
import os

from langchain_core.tools.base import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from src.agents.base import BaseAgent
from src.agents.state import State
from src.agents.lecturer.prompt import prompt_lecturer


class LecturerState(dict):
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

    async def _content_rewrite(self, state: LecturerState) -> LecturerState:
        return state

    async def _read_documents(self, state: LecturerState) -> LecturerState:
        document_path = state.get("document_path")
        if os.path.exists(document_path):
            try:
                loader = PyPDFLoader(document_path)
                documents = await loader.aload()
                state.update(documents=documents)
            except Exception as e:
                logging.exception(e)
        return state

    async def process(self, state: State, config: RunnableConfig) -> State:
        try:
            document_path = state.get("document_path")
            input_state = {
                "document_path": document_path,
                "documents": [],
                "next_node": "",
                "sermon": [],
                "url_image": [],
            }
            sub_graph = self.get_subgraph()
            response = await sub_graph.ainvoke(input=input_state)
            sermon = response.get("sermon")
            url_image = response.get("url_image")
            lecture = {
                url_image[i]: sermon[i] for i in range(min(len(sermon), len(url_image)))
            }
            state.update(lecture=lecture)
        except Exception as e:
            logging.exception(e)
        return state
