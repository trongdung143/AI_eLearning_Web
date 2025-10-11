import logging
import os
from typing import Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools.base import BaseTool, Field
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.agents.qa.prompt import (
    prompt_qa,
    prompt_supervisor,
    prompt_reviewer,
    prompt_question_rewrite,
)
from src.agents.state import State
from src.config.setup import GOOGLE_API_KEY, DATA_DIR


class QaState(dict):
    question: str
    genarate: str
    next_node: str
    document_path: str
    vectorstore_path: str
    documents: list[Document]


class SupervisorResponseFormat(BaseModel):
    # content: str = Field(description="Feedback")
    binary_score: str = Field(description="yes or no")


class ReviewerResponseFormat(BaseModel):
    # content: str = Field(description="NAH")
    binary_score: str = Field(description="yes or no")


class Supervisor(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="supervisor",
            state=QaState,
        )

        self._prompt = prompt_supervisor

        self._chain = self._prompt | self._model.with_structured_output(
            SupervisorResponseFormat
        )

    async def process(self, state: QaState) -> QaState:
        try:
            genarate = state.get("genarate")
            question = state.get("question")
            response = await self._chain.ainvoke(
                {"question": question, "genarate": genarate}
            )
            binary_score = response.binary_score
            next_node = "__end__" if binary_score == "yes" else "genarate"
            state.update(next_node=next_node)
        except Exception as e:
            logging.exception(e)
        return state


class Reviewer(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="reviewer",
            state=QaState,
        )

        self._prompt = prompt_reviewer

        self._chain = self._prompt | self._model.with_structured_output(
            ReviewerResponseFormat
        )

    async def process(self, state: QaState) -> QaState:
        try:
            documents = state.get("documents")
            question = state.get("question")
            doc_txt = documents[0].page_content
            response = await self._chain.ainvoke(
                {"question": question, "document": doc_txt}
            )
            binary_score = response.binary_score
            next_node = "genarate" if binary_score == "yes" else "re_question"
            state.update(next_node=next_node)
        except Exception as e:
            logging.exception(e)
        return state


class QuestionReWrite(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="question_rewrite",
            state=QaState,
        )

        self._prompt = prompt_question_rewrite

        self._chain = self._prompt | self._model.with_structured_output(
            ReviewerResponseFormat
        )

    async def process(self, state: QaState) -> QaState:
        try:
            question = state.get("question")
            response = await self._chain.ainvoke({"question": question})
            state.update(question=response.content)
        except Exception as e:
            logging.exception(e)
        return state


class QaAgent(BaseAgent):
    VALID_NODES = ["re_question", "genarate", "__end__"]

    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="qa",
            tools=tools,
            state=QaState,
        )

        self._prompt = prompt_qa

        self._chain = self._prompt | self._model

        self._supervisor = Supervisor()

        self._reviewer = Reviewer()

        self._question_rewrite = QuestionReWrite()

        self._embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )

        self._set_subgraph()

    def _format_document(self, state: QaState) -> str:
        documents = state.get("documents")
        full_txt = ""
        for doc in documents:
            full_txt = full_txt + doc.page_content + "\n\n"
        return full_txt

    def _route(self, state: QaState) -> str:
        next_node = state.get("next_node").strip()
        if next_node in self.VALID_NODES:
            return next_node
        return "__end__"

    def _set_subgraph(self):

        self._sub_graph.add_node("docs_to_vec", self._document_to_vector)
        self._sub_graph.add_node("retrieve", self._retrieve)
        self._sub_graph.add_node("reviewer", self._reviewer.process)
        self._sub_graph.add_node("re_question", self._question_rewrite.process)
        self._sub_graph.add_node("supervisor", self._supervisor.process)
        self._sub_graph.add_node("genarate", self._genarate)

        self._sub_graph.set_entry_point("docs_to_vec")

        self._sub_graph.add_edge("docs_to_vec", "retrieve")
        self._sub_graph.add_edge("retrieve", "reviewer")

        self._sub_graph.add_conditional_edges(
            "reviewer",
            self._route,
            {
                "re_question": "re_question",
                "genarate": "genarate",
            },
        )

        self._sub_graph.add_edge("re_question", "retrieve")
        self._sub_graph.add_edge("genarate", "supervisor")

        self._sub_graph.add_conditional_edges(
            "supervisor",
            self._route,
            {
                "__end__": "__end__",
                "genarate": "genarate",
            },
        )

    def _load_retriever(self, state: QaState) -> VectorStoreRetriever:
        vectorstore_path = state.get("vectorstore_path")
        if os.path.exists(vectorstore_path):
            try:
                vectorstore = FAISS.load_local(
                    folder_path=vectorstore_path,
                    embeddings=self._embedding,
                    allow_dangerous_deserialization=True,
                )
                retriever = vectorstore.as_retriever(
                    search_type="similarity", search_kwargs={"k": 4}
                )
                return retriever
            except Exception as e:
                logging.exception(e)
        return None

    async def _document_to_vector(self, state: QaState) -> QaState:
        document_path = state.get("document_path")
        vectorstore_path = state.get("vectorstore_path")
        if not os.path.exists(vectorstore_path):
            if os.path.exists(document_path):
                try:
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
                    logging.exception(e)
        return state

    async def _retrieve(self, state: QaState) -> QaState:
        try:

            question = state.get("question")
            retriever = self._load_retriever(state)
            response = await retriever.ainvoke(question)
            state.update(documents=response)
        except Exception as e:
            logging.exception(e)
        return state

    async def _genarate(self, state: QaState) -> QaState:
        try:
            question = state.get("question")
            full_txt = self._format_document(state)
            response = await self._chain.ainvoke(
                {"context": full_txt, "question": question}
            )
            genarate = response.content
            state.update(genarate=genarate)
        except Exception as e:
            logging.exception(e)
        return state

    async def process(self, state: State, config: RunnableConfig) -> State:
        try:
            question = state.get("task")
            lesson_id = config.get("configurable").get("lesson_id")

            document_path = f"{DATA_DIR}/pdf/{lesson_id}.pdf"
            vectorstore_path = f"{DATA_DIR}/vectorstore/{lesson_id}"

            input_state = {
                "question": question,
                "genarate": "",
                "next_node": "",
                "document_path": document_path,
                "vectorstore_path": vectorstore_path,
                "documents": [],
            }

            sub_graph = self.get_subgraph()
            response = await sub_graph.ainvoke(input=input_state)
            state.update(result=response.get("genarate"))
        except Exception as e:
            logging.exception(e)
        return state
