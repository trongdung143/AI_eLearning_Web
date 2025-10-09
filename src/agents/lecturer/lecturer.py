from typing import Sequence
import logging
import os

from langchain_core.tools.base import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools.base import BaseTool, Field
from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.agents.state import State
from src.agents.lecturer.prompt import (
    prompt_lecturer_first,
    prompt_lecturer_continue,
    prompt_reviewer,
)


class LecturerState(dict):
    next_node: str
    document_path: str
    documents: list[Document]
    sermon: list[str]
    url_image: list[str]
    document: Document | None
    prev_lecture: str | None
    lecture: str | None
    feedback: str
    index: int


class ReviewerResponseFormat(BaseModel):
    feedback: str = Field(description="your explanation and constructive comments")
    binary_score: str = Field(description="yes or no")


class Reviewer(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="reviewer",
            state=LecturerState,
        )

        self._prompt = prompt_reviewer

        self._chain = self._prompt | self._model.with_structured_output(
            ReviewerResponseFormat
        )

    def _format_document(self, document: Document) -> str:
        return document.page_content

    async def process(self, state: LecturerState) -> LecturerState:
        try:

            document = state.get("document")
            txt = self._format_document(document)
            lecture = state.get("lecture")
            prev_lecture = lecture
            response = self._chain.ainvoke({"document": txt, "lecture": lecture})
            binary_score = response.binary_score
            feedback = f"Ban vua viet {lecture} \n\n feedback {response.feedback}"
            next_node = None
            if binary_score == "no":
                next_node = "lecture"
            else:
                next_node = "receive_document"
                prev_lecture = lecture
            state.update(
                next_node=next_node, feedback=feedback, prev_lecture=prev_lecture
            )

        except Exception as e:
            logging.exception(e)
        return state


class LecturerAgent(BaseAgent):
    VALID_NODES = ["lecture", "receive_document"]

    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="lecturer",
            state=LecturerState,
            tools=tools,
        )

        self._reviewer = Reviewer()

        self._set_subgraph()

    def _route(self, state: LecturerState) -> str:
        next_node = state.get("next_node").strip()
        if next_node in self.VALID_NODES:
            return next_node
        return "__end__"

    def _receive_document(self, state: LecturerState) -> LecturerState:
        documents = state.get("documents")
        if index == 0:
            self._chain = prompt_lecturer_first | self._model
        else:
            self._chain = prompt_lecturer_continue | self._model

        if index >= len(documents):
            next_node = "__end__"
            state.update(next_node=next_node)
        else:
            next_node = "lecture"
            index = state.get("index")
            document = documents[index]
            state.update(document=document, index=index + 1, next_node=next_node)
        return state

    def _set_subgraph(self):
        self._sub_graph.add_node("read_documents", self._read_documents)
        self._sub_graph.add_node("receive_document", self._receive_document)
        self._sub_graph.add_node("lecture", self._lecture)
        self._sub_graph.add_node("reviewer", self._reviewer.process)

        self._sub_graph.set_entry_point("read_documents")
        self._sub_graph.add_edge("read_documents", "receive_document")
        self._sub_graph.add_conditional_edges(
            "receive_document",
            self._route,
            {
                "lecture": "lecture",
                "__end__": "__end__",
            },
        )

        self._sub_graph.add_edge("lecture", "reviewer")

        self._sub_graph.add_conditional_edges(
            "reviewer",
            self._route,
            {
                "lecture": "lecture",
                "receive_document": "receive_document",
            },
        )

    def _format_document(self, document: Document) -> str:
        return document.page_content

    async def _lecture(self, state: LecturerState) -> LecturerState:
        feedback = state.get("feedback")
        document = state.get("document")
        txt = self._format_document(document)

        index = state.get("index")
        response = None

        if index == 0:
            response = await self._chain.ainvoke(
                {"current_content": txt, "feedback": feedback}
            )
        else:
            prev_lecture = state.get("prev_lecture")

            response = await self._chain.ainvoke(
                {
                    "current_content": txt,
                    "previous_content": prev_lecture,
                    "feedback": feedback,
                }
            )

        lecture = response.content
        state.update(lecture=lecture)
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
                "document": None,
                "prev_lecture": "",
                "lecture": "",
                "feedback": "",
                "index": 0,
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
