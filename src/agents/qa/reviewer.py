from src.agents.qa.prompt import prompt_reviewer
from src.agents.qa.qa_state import QaState
from src.agents.qa.qa_state import ReviewerResponseFormat
from src.agents.utils import logger
from src.agents.base import BaseAgent


class Reviewer(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="reviewer", state=QaState)

        self._prompt = prompt_reviewer

        self._chain = self._prompt | self._model.with_structured_output(
            ReviewerResponseFormat
        )

    def _format_document(self, state: QaState) -> str:
        documents = state.get("documents")
        full_txt = ""
        for doc in documents:
            full_txt += doc.page_content + "\n\n"
        return full_txt

    async def process(self, state: QaState) -> QaState:
        try:
            question = state.get("question")

            doc_txt = self._format_document(state)

            response = await self._chain.ainvoke(
                {"question": question, "document": doc_txt}
            )

            binary_score = getattr(response, "binary_score", "no")
            feedback = ""
            next_node = ""

            if binary_score == "no":
                feedback = getattr(response, "feedback", "")
                next_node = "re_question"
            else:
                next_node = "generate"

        except Exception as e:
            logger.exception(e)
        finally:
            state.update(next_node=next_node, bs_rv=binary_score, feedback_rv=feedback)
            logger.info("[Reviewer] process executed")
        return state
