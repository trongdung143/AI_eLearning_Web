from src.agents.qa.prompt import prompt_writer
from src.agents.qa.qa_state import QaState
from src.agents.utils import logger
from src.agents.base import BaseAgent


class Writer(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_name="writer", state=QaState)

        self._prompt = prompt_writer

        self._chain = self._prompt | self._model

    async def process(self, state: QaState) -> QaState:
        try:
            generate = state.get("generate")
            question = state.get("question")
            response = await self._chain.ainvoke(
                {"question": question, "generate": generate}
            )
            answer = response.content

        except Exception as e:
            logger.exception(e)
        finally:
            state.update(answer=answer)
            logger.info("[Write] process executed")
        return state
