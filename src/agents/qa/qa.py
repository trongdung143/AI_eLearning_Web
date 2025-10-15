import os
from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.runnables.config import RunnableConfig
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage
from langchain_core.tools.base import BaseTool

from src.agents.base import BaseAgent
from src.agents.qa.prompt import prompt_qa
from src.agents.state import State
from src.config.setup import GOOGLE_API_KEY, DATA_DIR
from src.agents.qa.qa_state import QaState
from src.agents.qa.supervisor import Supervisor
from src.agents.qa.writer import Writer
from src.agents.qa.reviewer import Reviewer
from src.agents.qa.rewrite_question import QuestionReWrite

from src.agents.utils import logger


class QaAgent(BaseAgent):
    VALID_NODES = ["re_question", "generate", "__end__", "writer"]

    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(agent_name="qa", tools=tools, state=QaState)

        self._prompt = prompt_qa

        self._chain = self._prompt | self._model

        self._supervisor = Supervisor()

        self._reviewer = Reviewer()

        self._writer = Writer()

        self._question_rewrite = QuestionReWrite()

        self._embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY
        )

        self._set_subgraph()

    def _format_document(self, state: QaState) -> str:
        documents = state.get("documents")
        full_txt = ""
        for doc in documents:
            full_txt += doc.page_content + "\n\n"
        return full_txt

    def _route(self, state: QaState) -> str:
        next_node = state.get("next_node").strip()
        if next_node in self.VALID_NODES:
            return next_node
        return "__end__"

    def _set_subgraph(self):
        try:
            self._sub_graph.add_node("retrieve", self._retrieve)
            self._sub_graph.add_node("reviewer", self._reviewer.process)
            self._sub_graph.add_node("re_question", self._question_rewrite.process)
            self._sub_graph.add_node("supervisor", self._supervisor.process)
            self._sub_graph.add_node("generate", self._genarate)
            self._sub_graph.add_node("writer", self._writer.process)

            self._sub_graph.set_entry_point("retrieve")
            self._sub_graph.add_edge("retrieve", "reviewer")

            self._sub_graph.add_conditional_edges(
                "reviewer",
                self._route,
                {"re_question": "re_question", "generate": "generate"},
            )

            self._sub_graph.add_edge("re_question", "retrieve")
            self._sub_graph.add_edge("generate", "supervisor")

            self._sub_graph.add_conditional_edges(
                "supervisor",
                self._route,
                {"writer": "writer", "generate": "generate"},
            )

            self._sub_graph.set_finish_point("writer")
        except Exception as e:
            logger.exception(e)
        finally:
            logger.info("[QaAgent] _set_subgraph executed")

    def _load_retriever(self, state: QaState) -> VectorStoreRetriever:
        try:
            vectorstore_path = state.get("vectorstore_path")
            if os.path.exists(vectorstore_path):
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
            logger.exception(e)
        finally:
            logger.info("[QaAgent] _load_retriever executed")
        return None

    async def _retrieve(self, state: QaState) -> QaState:
        try:
            question = state.get("question")
            retriever = self._load_retriever(state)

            if retriever:
                response = await retriever.ainvoke(question)

        except Exception as e:
            logger.exception(e)
        finally:
            state.update(documents=response)
            logger.info("[QaAgent] _retrieve executed")
        return state

    async def _genarate(self, state: QaState) -> QaState:
        try:
            question = state.get("question")
            full_txt = self._format_document(state)
            feedback_sp = state.get("feedback_sp")

            response = await self._chain.ainvoke(
                {"context": full_txt, "question": question, "feedback": feedback_sp}
            )

            generate = response.content

        except Exception as e:
            logger.exception(e)
        finally:
            state.update(generate=generate)
            logger.info("[QaAgent] _genarate executed")
        return state

    async def process(self, state: State, config: RunnableConfig) -> State:
        try:
            question = state.get("task")
            lesson_id = config.get("configurable").get("lesson_id")
            vectorstore_path = f"{DATA_DIR}/vectorstore/{lesson_id}"
            input_state = {"question": question, "vectorstore_path": vectorstore_path}
            sub_graph = self.get_subgraph()

            response = await sub_graph.ainvoke(input=input_state)

            answer = response.get("answer")
        except Exception as e:
            logger.exception(e)
        finally:
            state.update(result=answer, messages=[AIMessage(content=answer)])
            logger.info("[QaAgent] process executed")
        return state
