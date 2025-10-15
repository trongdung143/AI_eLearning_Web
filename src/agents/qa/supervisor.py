from src.agents.qa.prompt import prompt_supervisor
from src.agents.qa.qa_state import QaState
from src.agents.qa.qa_state import SupervisorResponseFormat
from src.agents.utils import logger
from src.agents.base import BaseAgent


class Supervisor(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="supervisor", state=QaState)

        self._prompt = prompt_supervisor

        self._chain = self._prompt | self._model.with_structured_output(
            SupervisorResponseFormat
        )

    def _format_document(self, state: QaState) -> str:
        documents = state.get("documents")
        full_txt = ""
        for doc in documents:
            full_txt += doc.page_content + "\n\n"
        return full_txt

    async def process(self, state: QaState) -> QaState:
        try:
            generate = state.get("generate")
            question = state.get("question")
            doc_txt = self._format_document(state)

            response = await self._chain.ainvoke(
                {"question": question, "generate": generate, "context": doc_txt}
            )

            binary_score = getattr(response, "binary_score", "no")
            feedback = ""
            next_node = ""

            if binary_score == "no":
                next_node = "generate"
                feedback = getattr(response, "feedback", "")
            else:
                next_node = "writer"

        except Exception as e:
            logger.exception(e)
        finally:
            state.update(next_node=next_node, feedback_sp=feedback, bs_sp=binary_score)
            logger.info("[Supervisor] process executed")
        return state
