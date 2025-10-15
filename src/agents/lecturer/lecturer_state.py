from langchain_core.documents import Document
from langchain_core.tools.base import Field
from pydantic import BaseModel


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
