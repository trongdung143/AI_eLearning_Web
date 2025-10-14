from typing import Sequence
import logging
import os
import re
import json
import shutil

from langchain_core.tools.base import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools.base import BaseTool, Field
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    GOOGLE_API_KEY,
)

import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=CLOUDINARY_API_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True,
)

logger = logging.getLogger(__name__)


class LecturerState(dict):
    next_node: str = ""
    document_path: str
    vectorstore_path: str
    slide_dir: str = ""
    documents: list[Document] = Field(default_factory=list)
    lectures: list[str] = Field(default_factory=list)
    lectures_segments: list[list[str]] = Field(default_factory=list)
    slide_urls: list[str] = Field(default_factory=list)
    current_page: Document | None
    prev_lecture: str = ""
    current_lecture: str = ""
    feedback: str = ""
    binary_score: str = ""
    lesson_id: str
    page_index: int


class ReviewerResponseFormat(BaseModel):
    feedback: str = Field(description="your explanation and constructive comments")
    binary_score: str = Field(description="yes or no")


class LecturerSegmentResponseFormat(BaseModel):
    segment: str = Field(description="is a string with an array")


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

    def _format_document(self, current_page: Document) -> str:
        return current_page.page_content

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
            lectures = state.get("lectures", [])
            current_page = state.get("current_page")
            current_lecture = state.get("current_lecture", "")
            prev_lecture = state.get("prev_lecture", "")

            next_node = "lectures_segments"
            feedback = ""
            txt = self._format_document(current_page)

            response = await self._chain.ainvoke(
                {
                    "current_page": txt,
                    "current_lecture": current_lecture,
                    "previous_lecture": prev_lecture,
                }
            )

            binary_score = getattr(response, "binary_score", "no")
            feedback = getattr(response, "feedback", "")
            logger.info(f"[LecturerAgent] Review result: {binary_score}")

            if binary_score == "no":
                next_node = "generate_lecture"
                feedback = f"Lời giảng bạn vừa viết:\n{current_lecture}\n\nFeedback:\n{feedback}"
            else:
                next_node = "lectures_segments"
                feedback = ""
                current_lecture = self._clean_txt(current_lecture)
                prev_lecture = current_lecture
                lectures.append(current_lecture)

        except Exception as e:
            logger.exception(
                f"[LecturerAgent] Error while processing page {current_page}: {e}"
            )

        finally:
            state.update(
                next_node=next_node,
                feedback=feedback,
                prev_lecture=prev_lecture,
                lectures=lectures,
                current_lecture=current_lecture,
                binary_score=binary_score,
            )

        return state


class LecturerSegmentAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            agent_name="lectures_segments",
            state=LecturerState,
        )

        self._prompt = prompt_lecturer_segment

        self._chain = self._prompt | self._model

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
            lectures_segments = state.get("lectures_segments", [])
            current_lecture = state.get("current_lecture", "")
            prev_lecture = state.get("prev_lecture", "")
            clean_lecture_segment = []
            response = await self._chain.ainvoke(
                {"previous_lecture": prev_lecture, "current_lecture": current_lecture}
            )

            try:
                raw_content = getattr(response, "content", "")

                raw_content = (
                    raw_content.replace("```json", "").replace("```", "").strip()
                )

                lecture_segment = json.loads(raw_content)

                clean_lecture_segment = [
                    self._clean_txt(seg).strip()
                    for seg in lecture_segment.get("segment")
                    if isinstance(seg, str) and seg.strip()
                ]
                logger.info("[LecturerAgent] Lecture segment parsed successfully")
            except Exception as e:
                logger.exception(f"[LecturerAgent] Invalid JSON response: {e}")

        except Exception as e:
            logger.exception(f"[LecturerAgent] Error processing lecture segment: {e}")

        finally:
            if clean_lecture_segment:
                lectures_segments.append(clean_lecture_segment)
            state.update(lectures_segments=lectures_segments)
        return state


class LecturerAgent(BaseAgent):
    VALID_NODES = [
        "generate_lecture",
        "receive_document",
        "upload_document",
        "lectures_segments",
        "document_to_vector",
    ]

    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="lecturer",
            state=LecturerState,
            tools=tools,
        )

        self._reviewer = Reviewer()

        self._embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )

        self._lecturer_segment = LecturerSegmentAgent()

        self._set_subgraph()

    def _route(self, state: LecturerState) -> str:
        next_node = state.get("next_node").strip()
        if next_node in self.VALID_NODES:
            return next_node
        return "__end__"

    def _receive_document(self, state: LecturerState) -> LecturerState:
        try:
            documents = state.get("documents", [])
            page_index = state.get("page_index")
            if page_index == 0:
                self._chain = prompt_lecturer_first | self._model
            else:
                self._chain = prompt_lecturer_continue | self._model

            if page_index >= len(documents):
                next_node = "document_to_vector"

            else:
                next_node = "generate_lecture"
                current_page = documents[page_index]

        except Exception as e:
            logger.exception(e)
        finally:
            if page_index >= len(documents):
                state.update(next_node=next_node)
            else:
                state.update(
                    current_page=current_page,
                    page_index=page_index + 1,
                    next_node=next_node,
                )
            logger.info("[LecturerAgent] _receive_document executed")
        return state

    def _upload_document(self, state: LecturerState) -> LecturerState:
        try:
            page_index = state.get("page_index")
            lesson_id = state.get("lesson_id")
            file_base = state.get("slide_dir")
            file_path = f"{file_base}/slide_{page_index}.pdf"
            slide_urls = state.get("slide_urls", [])

            upload_result = cloudinary.uploader.upload(
                file_path,
                resource_type="raw",
                public_id=f"slide_{page_index}",
                folder=lesson_id,
                overwrite=True,
            )

            secure_url = upload_result.get("secure_url")
            slide_urls.append(secure_url)

        except Exception as e:
            logger.exception(e)
        finally:
            state.update(slide_urls=slide_urls)
            logger.info("[LecturerAgent] _upload_document executed")
        return state

    async def _document_to_vector(self, state: LecturerState) -> LecturerState:
        try:
            document_path = state.get("document_path")
            vectorstore_path = state.get("vectorstore_path")
            base_path = "src/data/vectorstore"

            if not os.path.exists(vectorstore_path):
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                if os.path.exists(document_path):
                    loader = PyPDFLoader(document_path)
                    documents = loader.load()
                    text_splitter = (
                        RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                            chunk_size=500, chunk_overlap=100
                        )
                    )
                    documents_splits = text_splitter.split_documents(documents)

                    vectorstore = await FAISS.afrom_documents(
                        documents=documents_splits,
                        embedding=self._embedding,
                    )
                    vectorstore.save_local(vectorstore_path)
        except Exception as e:
            logger.exception(e)
        finally:
            logger.info("[LecturerAgent] _document_to_vector executed")
        return state

    def _set_subgraph(self):
        try:
            self._sub_graph.add_node("read_documents", self._read_documents)
            self._sub_graph.add_node("receive_document", self._receive_document)
            self._sub_graph.add_node("generate_lecture", self._genarate_lecture)
            self._sub_graph.add_node("reviewer", self._reviewer.process)
            self._sub_graph.add_node("split_document", self._split_document)
            self._sub_graph.add_node("upload_document", self._upload_document)
            self._sub_graph.add_node(
                "lectures_segments", self._lecturer_segment.process
            )
            self._sub_graph.add_node("document_to_vector", self._document_to_vector)

            self._sub_graph.set_entry_point("split_document")
            self._sub_graph.add_edge("split_document", "read_documents")
            self._sub_graph.add_edge("read_documents", "receive_document")
            self._sub_graph.add_conditional_edges(
                "receive_document",
                self._route,
                {
                    "generate_lecture": "generate_lecture",
                    "document_to_vector": "document_to_vector",
                },
            )
            self._sub_graph.add_edge("generate_lecture", "reviewer")
            self._sub_graph.add_conditional_edges(
                "reviewer",
                self._route,
                {
                    "generate_lecture": "generate_lecture",
                    "lectures_segments": "lectures_segments",
                },
            )
            self._sub_graph.add_edge("lectures_segments", "upload_document")
            self._sub_graph.add_edge("upload_document", "receive_document")
            self._sub_graph.set_finish_point("document_to_vector")
        except Exception as e:
            logger.exception(e)
        finally:
            logger.info("[LecturerAgent] _set_subgraph executed")

    def _format_document(self, current_page) -> str:
        return current_page.page_content

    def _split_document(self, state: LecturerState) -> LecturerState:
        try:
            document_path = state.get("document_path")
            lesson_id = state.get("lesson_id")
            output_dir = f"{DATA_DIR}/slide/{lesson_id}"
            reader = PdfReader(document_path)
            total_pages = len(reader.pages)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            for i in range(total_pages):
                writer = PdfWriter()
                writer.add_page(reader.pages[i])
                output_path = os.path.join(output_dir, f"slide_{i+1}.pdf")
                with open(output_path, "wb") as f:
                    writer.write(f)

        except Exception as e:
            logger.exception(e)
        finally:

            state.update(slide_dir=output_dir)
            logger.info("[LecturerAgent] _split_document executed")
        return state

    @traceable
    async def _genarate_lecture(self, state: LecturerState) -> LecturerState:
        try:
            feedback = state.get("feedback")
            current_page = state.get("current_page")
            txt = self._format_document(current_page)
            page_index = state.get("page_index")
            response = None

            if page_index == 0:
                response = await self._chain.ainvoke(
                    {"current_content": txt, "feedback": feedback}
                )
            else:
                prev_lecture = state.get("prev_lecture")
                response = await self._chain.ainvoke(
                    {
                        "current_content": txt,
                        "previous_lecture": prev_lecture,
                        "feedback": feedback,
                    }
                )

            current_lecture = response.content

        except Exception as e:
            logger.exception(e)
        finally:
            state.update(current_lecture=current_lecture)
            logger.info("[LecturerAgent] _genarate_lecture executed")
        return state

    async def _read_documents(self, state: LecturerState) -> LecturerState:
        try:
            document_path = state.get("document_path")
            if os.path.exists(document_path):
                loader = PyPDFLoader(document_path)
                documents = await loader.aload()

        except Exception as e:
            logger.exception(e)
        finally:
            state.update(documents=documents)
            logger.info("[LecturerAgent] _read_documents executed")
        return state

    async def process(self, state: State, config: RunnableConfig) -> State:
        try:
            document_path = state.get("document_path")
            lesson_id = state.get("lesson_id")
            vectorstore_path = f"{DATA_DIR}/vectorstore/{lesson_id}"

            input_state = {
                "document_path": document_path,
                "vectorstore_path": vectorstore_path,
                "page_index": 0,
                "lesson_id": lesson_id,
            }

            sub_graph = self.get_subgraph()
            response = await sub_graph.ainvoke(
                input=input_state, config={"recursion_limit": 50}
            )

            lectures = response.get("lectures")
            lectures_segments = response.get("lectures_segments")
            slide_urls = response.get("slide_urls")

            lecture = {
                slide_urls[i]: (lectures[i], lectures_segments[i])
                for i in range(len(slide_urls))
            }

        except Exception as e:
            logger.exception(e)
        finally:
            if document_path and os.path.exists(document_path):
                try:
                    os.remove(document_path)
                except Exception as e:
                    logger.exception(f"Could not remove document: {e}")

            slide_dir = f"src/data/slide/{lesson_id}"
            if os.path.exists(slide_dir):
                try:
                    shutil.rmtree(slide_dir)
                except Exception as e:
                    logger.exception(f"Could not remove slide folder: {e}")
            state.update(lecture=lecture)
            logger.info("[LecturerAgent] process executed")
        return state
