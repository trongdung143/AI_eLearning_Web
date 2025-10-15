from langchain_core.documents import Document
from langchain_core.tools.base import Field
from pydantic import BaseModel


class QaState(dict):
    feedback_sp: str = ""
    feedback_rv: str = ""
    question: str
    generate: str = ""
    next_node: str = ""
    vectorstore_path: str
    documents: list[Document] = Field(default_factory=list)
    bs_sp: str = ""
    bs_rv: str = ""
    answer: str = ""


class SupervisorResponseFormat(BaseModel):
    feedback: str = Field(
        description="a short but clear explanation or suggestion for improvement"
    )
    binary_score: str = Field(description="yes or no")


class ReviewerResponseFormat(BaseModel):
    binary_score: str = Field(description="yes or no")
    feedback: str = Field(
        description="short feedback explaining your reasoning OR suggesting a clearer rewritten version of the question"
    )
