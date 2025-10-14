from fastapi import APIRouter, UploadFile, File, Form
import uuid
from src.agents.workflow import graph
from src.config.setup import DATA_DIR
import os

router = APIRouter()

PDF_DIR = f"{DATA_DIR}/pdf"


@router.post("/lecturer")
async def qa_stream(
    course_id: str = Form(""),
    pdf_file: UploadFile = File(...),
) -> dict:

    lesson_id = str(uuid.uuid4()).strip()
    temp_path = f"{PDF_DIR}/{lesson_id}.pdf"

    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)

    with open(temp_path, "wb") as f:
        content = await pdf_file.read()
        f.write(content)

    config = {
        "configurable": {
            "thread_id": lesson_id,
            "lesson_id": lesson_id,
            "course_id": course_id,
        }
    }
    input_state = {
        "type_request": "lecturer",
        "task": None,
        "result": None,
        "lecture": None,
        "quiz": None,
        "document_path": temp_path,
        "lesson_id": lesson_id,
    }

    response = await graph.ainvoke(input=input_state, config=config)

    lecture = response.get("lecture")
    content_list = [
        {
            "url_pdf": url_pdf,
            "lecturer": lecturer,
            "lecturer_segment": lecturer_segment,
        }
        for url_pdf, (lecturer, lecturer_segment) in lecture.items()
    ]

    return {"lesson_id": lesson_id, "content": content_list}
