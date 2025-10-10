from fastapi import APIRouter, UploadFile, File, Form
import uuid
from src.agents.workflow import graph

router = APIRouter()

TEMP_DIR = "src/data/pdf"


@router.post("/lecturer")
async def qa_stream(
    type_request: str = Form(""),
    course_id: str = Form(""),
    pdf_file: UploadFile = File(...),
) -> dict:

    lesson_id = str(uuid.uuid4()).strip()
    temp_path = f"{TEMP_DIR}/{lesson_id}.pdf"

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
        "type_request": type_request,
        "task": None,
        "result": None,
        "lecture": None,
        "quiz": None,
        "document_path": temp_path,
        "lesson_id": lesson_id,
    }

    response = await graph.ainvoke(input=input_state, config=config)

    lecture = response.get("lecture")
    content_list = []

    for url_pdf, page_text in lecture.items():
        content_list.append({"url_pdf": url_pdf, "content": page_text})

    return {"lesson_id": lesson_id, "content": content_list}
