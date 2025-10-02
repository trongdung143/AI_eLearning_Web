from typing import Sequence
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.tools.base import BaseTool, Field
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.agents.rag.prompt import prompt_rag, prompt_supervisor, prompt_reviewer
from src.agents.state import State
from src.agents.supervisor.supervisor import SupervisorAgent
from src.config.setup import GOOGLE_API_KEY


class RagState(dict):
    task: str
    result: str
    next_node: str
    document_path: str
    vectorstore_path: str
    retriever: list[str] | None = None

class SupervisorResponseFormatForRag(BaseModel):
    content: str = Field(description="feedback")
    binary_score: str = Field(description="yes or no")

class SupervisorForRag(SupervisorAgent):
    def __init__(self):
        super().__init__(agent_name="supervisor", state=RagState, prompt=prompt_supervisor, response_format=SupervisorResponseFormatForRag)

    async def process(self, state: RagState) -> RagState:
        result = state.get("result")
        task = state.get("task")
        response = await self._chain.ainvoke({"task": [HumanMessage(content=f"### câu hỏi\n{task}\n\n### Câu trả lời\n{result}")]})
        if response.binary_score == "yes":
            next_node = "__end__"
        else:
            next_node = "genarate"
        state.update(next_node=next_node)
        return state

class ReviewerResponseFormatForRag(BaseModel):
    content: str = Field(description="NAH")
    binary_score: str = Field(description="yes or no")

class ReviewerForRag(SupervisorAgent):
    def __init__(self):
        super().__init__(agent_name="reviewer", state=RagState, prompt=prompt_reviewer, response_format=ReviewerResponseFormatForRag)

    async def process(self, state: RagState) -> RagState:
        retriever = state.get("retriever")
        filtered_docs = []
        next_node = "genarate"
        for doc in retriever:
            response = await self._chain.ainvoke({"task": [HumanMessage(content=f"### Tài liệu\n{doc}\n\n### Hãy đánh giá")]})
            if response.binary_score == "yes":
               filtered_docs.append(doc)
        if len(filtered_docs) == 0:
            next_node = "re_question"

        state.update(next_node=next_node, retriever=filtered_docs)
        return state

class RagAgent(BaseAgent):
    VALID_NODES = [
        "re_question",
        "genarate",
        "__end__"
    ]
    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="rag",
            tools=tools,
            state=RagState,
            model=None,
        )

        self._prompt = prompt_rag

        self._chain = self._prompt | self._model

        self._retriever = None

        self._supervisor = SupervisorForRag()

        self._reviewer = ReviewerForRag()

        self._embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )

        self._set_subgraph()

    def _route(self, state: RagState) -> str:
        next_node = state.get("next_node").strip()
        if next_node in self.VALID_NODES:
            return next_node
        return "__end__"

    def _format_docs(self, state: RagState) -> str:
        docs = state.get("retriever")
        txt = ""
        for doc in docs:
            txt += f"{doc}\n"
        return txt

    def _set_subgraph(self):
        self._sub_graph.add_node("supervisor", self._supervisor.process)
        self._sub_graph.add_node("docs_to_vec", self._document_to_vector)
        self._sub_graph.add_node("retrieve", self._retrieve)
        self._sub_graph.add_node("reviewer", self._reviewer.process)
        self._sub_graph.add_node("genarate", self._genarate)
        self._sub_graph.add_node("re_question", self._re_question)


        self._sub_graph.set_entry_point("docs_to_vec")

        self._sub_graph.add_edge("docs_to_vec", "retrieve")
        self._sub_graph.add_edge("retrieve", "reviewer")

        self._sub_graph.add_conditional_edges(
            "reviewer",
            self._route,
            {
                "re_question": "re_question",
                "genarate": "genarate",
            }
        )

        self._sub_graph.add_edge("re_question", "retrieve")
        self._sub_graph.add_edge("genarate", "supervisor")

        self._sub_graph.add_conditional_edges(
            "supervisor",
            self._route,
            {
                "__end__" : "__end__",
                "genarate" : "genarate",
            }
        )



    async def _document_to_vector(self, state: RagState) -> RagState:
        document_path = state.get("document_path")
        vector_store_path = state.get("vectorstore_path")
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"PDF not found at: {document_path}")
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
        documents_splits = text_splitter.split_documents(documents)

        # vectorstore = await Chroma.afrom_documents(
        #     documents=documents_splits,
        #     collection_name=vector_store_path,
        #     embedding=self._embedding,
        #     persist_directory=vector_store_path,
        # )
        #
        # vectorstore.persist()

        vectorstore = await FAISS.afrom_documents(
            documents=documents_splits,
            embedding=self._embedding,
        )

        vectorstore.save_local(vector_store_path)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
                "score_threshold": 0.8,
                "fetch_k": 20,
                "lambda_mult": 0.5,
            }
        )

        self._retriever = retriever
        return state

    async def _retrieve(self, state: RagState) -> RagState:
        task = state.get("task")
        response = self._retriever.ainvoke(task)
        retriever = [doc.page_content for doc in response]
        state.update(retriever=retriever)
        return state

    async def _genarate(self, state: RagState) -> RagState:
        task = state.get("task")
        txt = self._format_docs(state)
        response = await self._chain.ainvoke({"task": [HumanMessage(content=f"### Câu hỏi{task}\n\n### Tài liệu\n{txt}")]})
        result = response.content
        state.update(result=result)
        return state

    async def _re_question(self, state: RagState) -> RagState:
        return state

    async def process(self, state: State) -> State:
        task = self.get_task(state)
        result = None
        input_state = {
            "task": task,
            "result": "",
            "conversation_id": state.get("thread_id"),
            "document_path":  "src/data/temp/test.pdf",
            "vectorstore_path": "src/data/temp/test" # + f"{state.get('thread_id')}",
        }
        sub_graph = self.get_subgraph()
        response = await sub_graph.ainvoke(input=input_state)
        result = response.content
        current_tasks, current_results, assigned_agents = self.update_work(state, task, result)
        state.update(
            human=False,
            prev_agent=self._agent_name,
            tasks=current_tasks,
            results=current_results,
            assigned_agents=assigned_agents,
        )
        return state
