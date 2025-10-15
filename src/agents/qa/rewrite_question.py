from src.agents.qa.prompt import prompt_question_rewrite
from src.agents.qa.qa_state import QaState
from src.agents.utils import logger
from src.agents.base import BaseAgent


class QuestionReWrite(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="question_rewrite", state=QaState)

        self._prompt = prompt_question_rewrite

        self._chain = self._prompt | self._model

    async def process(self, state: QaState) -> QaState:
        try:
            question = state.get("question")
            feedback_rv = state.get("feedback_rv")

            response = await self._chain.ainvoke(
                {"question": question, "feedback": feedback_rv}
            )

        except Exception as e:
            logger.exception(e)
        finally:
            state.update(question=response.content)
            logger.info("[QuestionReWrite] process executed")
        return state
