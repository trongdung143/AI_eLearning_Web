import json

from src.agents.base import BaseAgent
from src.agents.lecturer.lecturer_state import LecturerState
from src.agents.lecturer.prompt import prompt_lecturer_segment
from src.agents.utils import logger
from src.agents.utils import clean_txt


class LecturerSegment(BaseAgent):

    def __init__(self):
        super().__init__(
            agent_name="lectures_segments",
            state=LecturerState,
        )

        self._prompt = prompt_lecturer_segment

        self._chain = self._prompt | self._model

    async def process(self, state: LecturerState) -> LecturerState:

        try:
            lectures_segments = state.get("lectures_segments", [])
            current_lecture = state.get("current_lecture", "")
            prev_lecture = state.get("prev_lecture", "")
            clean_lecture_segment = []
            response = await self._chain.ainvoke(
                {"previous_lecture": prev_lecture, "current_lecture": current_lecture}
            )

            try:
                raw_content = getattr(response, "content", "")

                raw_content = (
                    raw_content.replace("```json", "").replace("```", "").strip()
                )

                lecture_segment = json.loads(raw_content)

                clean_lecture_segment = [
                    clean_txt(seg).strip()
                    for seg in lecture_segment.get("segment")
                    if isinstance(seg, str) and seg.strip()
                ]
                logger.info("[LecturerAgent] Lecture segment parsed successfully")
            except Exception as e:
                logger.exception(f"[LecturerAgent] Invalid JSON response: {e}")

        except Exception as e:
            logger.exception(f"[LecturerAgent] Error processing lecture segment: {e}")

        finally:
            if clean_lecture_segment:
                lectures_segments.append(clean_lecture_segment)
            state.update(lectures_segments=lectures_segments)
        return state
