from typing import Sequence
import logging
import os
import re
import json

from langchain_core.tools.base import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools.base import BaseTool, Field
from langsmith import traceable
from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.agents.state import State
from src.agents.lecturer.prompt import (
    prompt_lecturer_first,
    prompt_lecturer_continue,
    prompt_reviewer,
    prompt_lecturer_segment,
)
from pypdf import PdfReader, PdfWriter

from src.config.setup import (
    CLOUDINARY_API_KEY,
    CLOUDINARY_API_NAME,
    CLOUDINARY_API_SECRET,
    DATA_DIR,
)

import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=CLOUDINARY_API_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True,
)


class LecturerState(dict):
    next_node: str
    document_path: str
    slide_path: str
    documents: list[Document]
    lecturer: list[str]
    lecturer_segment: list[list[str]]
    url_slide: list[str]
    document: Document | None
    prev_lecture: str | None
    lecture: str | None
    feedback: str | None
    lesson_id: str
    index: int


class ReviewerResponseFormat(BaseModel):
    feedback: str = Field(description="your explanation and constructive comments")
    binary_score: str = Field(description="yes or no")


class LecturerSegmentResponseFormat(BaseModel):
    content: str = Field(
        description="A list of short, natural, and coherent lecture segments"
    )


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

    def _clean_txt(self, txt: str) -> str:
        txt = (
            txt.replace("\\n", " ")
            .replace("\n", " ")
            .replace("\t", " ")
            .replace("\r", " ")
        )

        txt = re.sub(r"[^A-Za-zÀ-ỹ0-9\s]", " ", txt)

        txt = re.sub(r"\s+", " ", txt).strip()

        return txt

    async def process(self, state: LecturerState) -> LecturerState:
        try:
            lecturer = state.get("lecturer")
            document = state.get("document")
            txt = self._format_document(document)
            lecture = state.get("lecture")
            prev_lecture = state.get("prev_lecture")
            response = await self._chain.ainvoke({"document": txt, "lecture": lecture})
            binary_score = response.binary_score
            feedback = f"Lời giảng bạn vừa viết:\n{lecture} \n\n feedback:\n{response.feedback}"
            next_node = None

            if binary_score == "no":
                next_node = "lecturer"
            else:
                feedback = ""
                next_node = "lecturer_segment"
                lecture = self._clean_txt(lecture)
                prev_lecture = lecture
                lecturer.append(lecture)
            state.update(
                next_node=next_node,
                feedback=feedback,
                prev_lecture=prev_lecture,
                lecturer=lecturer,
                lecture=lecture,
            )

        except Exception as e:
            logging.exception(e)
        return state


class LecturerSegmentAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            agent_name="lecturer_segment",
            state=LecturerState,
        )

        self._prompt = prompt_lecturer_segment

        self._chain = self._prompt | self._model.with_structured_output(
            LecturerSegmentResponseFormat
        )

    def _clean_txt(self, txt: str) -> str:
        txt = (
            txt.replace("\\n", " ")
            .replace("\n", " ")
            .replace("\t", " ")
            .replace("\r", " ")
        )

        txt = re.sub(r"[^A-Za-zÀ-ỹ0-9\s]", " ", txt)

        txt = re.sub(r"\s+", " ", txt).strip()

        return txt

    async def process(self, state: LecturerState) -> LecturerState:
        try:
            lecturer_segment = state.get("lecturer_segment")
            lecture = state.get("lecture")
            prev_lecture = state.get("prev_lecture")
            lecturer_segment = state.get("lecturer_segment")
            response = await self._chain.ainvoke(
                {"previous_lecture": prev_lecture, "lecture": lecture}
            )
            lecture_segment = json.loads(response.content)

            clean_lecture_segment = [
                self._clean_txt(seg).strip() for seg in lecture_segment
            ]

            lecturer_segment.append(clean_lecture_segment)
            state.update(lecturer_segment=lecturer_segment)
        except Exception as e:
            logging.exception(e)
        return state


class LecturerAgent(BaseAgent):
    VALID_NODES = ["lecturer", "receive_document", "upload_pdf", "lecturer_segment"]

    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="lecturer",
            state=LecturerState,
            tools=tools,
        )

        self._reviewer = Reviewer()

        self._lecturer_segment = LecturerSegmentAgent()

        self._set_subgraph()

    def _route(self, state: LecturerState) -> str:
        next_node = state.get("next_node").strip()
        if next_node in self.VALID_NODES:
            return next_node
        return "__end__"

    def _receive_document(self, state: LecturerState) -> LecturerState:
        documents = state.get("documents")
        index = state.get("index")
        if index == 0:
            self._chain = prompt_lecturer_first | self._model
        else:
            self._chain = prompt_lecturer_continue | self._model

        if index >= len(documents):
            next_node = "__end__"
            state.update(next_node=next_node)
        else:
            next_node = "lecturer"
            index = state.get("index")
            document = documents[index]
            state.update(document=document, index=index + 1, next_node=next_node)
        return state

    def _upload_pdf(self, state: LecturerState) -> LecturerState:
        index = state.get("index")
        lesson_id = state.get("lesson_id")
        file_base = state.get("slide_path")
        file_path = f"{file_base}/slide_{index}.pdf"
        url_slide = state.get("url_slide")

        upload_result = cloudinary.uploader.upload(
            file_path,
            resource_type="raw",
            public_id=f"slide_{index}",
            folder=lesson_id,
            overwrite=True,
        )

        secure_url = upload_result.get("secure_url")
        url_slide.append(secure_url)
        state.update(url_slide=url_slide)
        return state

    def _set_subgraph(self):
        self._sub_graph.add_node("read_documents", self._read_documents)
        self._sub_graph.add_node("receive_document", self._receive_document)
        self._sub_graph.add_node("lecturer", self._lecturer)
        self._sub_graph.add_node("reviewer", self._reviewer.process)
        self._sub_graph.add_node("split_pdf", self._split_pdf)
        self._sub_graph.add_node("upload_pdf", self._upload_pdf)
        self._sub_graph.add_node("lecturer_segment", self._lecturer_segment.process)

        self._sub_graph.set_entry_point("split_pdf")
        self._sub_graph.add_edge("split_pdf", "read_documents")
        self._sub_graph.add_edge("read_documents", "receive_document")
        self._sub_graph.add_conditional_edges(
            "receive_document",
            self._route,
            {
                "lecturer": "lecturer",
                "__end__": "__end__",
            },
        )

        self._sub_graph.add_edge("lecturer", "reviewer")

        self._sub_graph.add_conditional_edges(
            "reviewer",
            self._route,
            {
                "lecturer": "lecturer",
                "lecturer_segment": "lecturer_segment",
            },
        )
        self._sub_graph.add_edge("lecturer_segment", "upload_pdf")
        self._sub_graph.add_edge("upload_pdf", "receive_document")

    def _format_document(self, document: Document) -> str:
        return document.page_content

    def _split_pdf(self, state: LecturerState) -> LecturerState:
        document_path = state.get("document_path")
        lesson_id = state.get("lesson_id")
        output_dir = f"{DATA_DIR}/slide/{lesson_id}"
        reader = PdfReader(document_path)
        total_pages = len(reader.pages)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(total_pages):
            writer = PdfWriter()
            writer.add_page(reader.pages[i])

            output_path = os.path.join(output_dir, f"slide_{i+1}.pdf")
            with open(output_path, "wb") as f:
                writer.write(f)

        state.update(slide_path=output_dir)
        return state

    @traceable
    async def _lecturer(self, state: LecturerState) -> LecturerState:
        feedback = state.get("feedback")
        document = state.get("document")
        txt = self._format_document(document)

        index = state.get("index")
        response = None

        try:
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
        except Exception as e:
            logging.exception(e)
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
            lesson_id = state.get("lesson_id")
            input_state = {
                "document_path": document_path,
                "slide_path": "",
                "documents": [],
                "next_node": "",
                "lecturer": [],
                "lecturer_segment": [],
                "url_slide": [],
                "document": None,
                "prev_lecture": "",
                "lecture": "",
                "feedback": None,
                "index": 0,
                "lesson_id": lesson_id,
            }
            sub_graph = self.get_subgraph()
            response = await sub_graph.ainvoke(input=input_state)
            lecturer = response.get("lecturer")
            lecturer_segment = response.get("lecturer_segment")
            url_slide = response.get("url_slide")
            lecture = {
                url_slide[i]: (lecturer[i], lecturer_segment[i])
                for i in range(len(url_slide))
            }
            state.update(lecture=lecture)
        except Exception as e:
            logging.exception(e)
        return state
