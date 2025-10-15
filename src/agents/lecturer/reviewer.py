from src.agents.base import BaseAgent
from src.agents.lecturer.lecturer_state import LecturerState, ReviewerResponseFormat
from src.agents.lecturer.prompt import prompt_reviewer
from src.agents.utils import logger
from src.agents.utils import clean_txt, format_document


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

    async def process(self, state: LecturerState) -> LecturerState:
        try:
            lectures = state.get("lectures", [])
            current_page = state.get("current_page")
            current_lecture = state.get("current_lecture", "")
            prev_lecture = state.get("prev_lecture", "")

            next_node = "lectures_segments"
            feedback = ""
            txt = format_document(current_page)

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
                current_lecture = clean_txt(current_lecture)
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
