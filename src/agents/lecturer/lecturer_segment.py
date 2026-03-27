import json

from src.agents.base import BaseAgent
from src.agents.lecturer.lecturer_state import LecturerState
from src.agents.lecturer.prompt import prompt_lecturer_segment
from src.agents.utils import logger
from src.agents.utils import clean_txt, extract_content


class LecturerSegment(BaseAgent):

    def __init__(self):
        super().__init__(
            agent_name="lectures_segments",
            state=LecturerState,
        )

        self._prompt = prompt_lecturer_segment

        self._chain = self._prompt | self._model

    async def process(self, state) -> "LecturerState":
        try:
            lectures_segments = state.get("lectures_segments", [])
            current_lecture = state.get("current_lecture", "")
            prev_lecture = state.get("prev_lecture", "")
            clean_lecture_segment = []

            current_lecture = clean_txt(current_lecture)
            prev_lecture = clean_txt(prev_lecture)

            response = await self._chain.ainvoke(
                {"previous_lecture": prev_lecture, "current_lecture": current_lecture}
            )

            try:
                raw_content = extract_content(response)

                if isinstance(raw_content, str):
                    raw_content = (
                        raw_content.replace("```json", "").replace("```", "").strip()
                    )

                lecture_segment = json.loads(raw_content)

                clean_lecture_segment = [
                    clean_txt(seg).strip() for seg in lecture_segment.get("segment", [])
                ]

                logger.info("[LecturerAgent] Lecture segment parsed successfully")

            except Exception as e:
                logger.exception(f"[LecturerAgent] Invalid JSON response: {e}")

            if clean_lecture_segment:
                lectures_segments.append(clean_lecture_segment)

            print(lecture_segment)
            state.update(lectures_segments=lectures_segments)

        except Exception as e:
            logger.exception(f"[LecturerAgent] Error processing lecture segment: {e}")

        finally:
            logger.info("[LecturerAgent] _process_lecture_segment executed")

        return state
