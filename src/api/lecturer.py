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

) -> tuple[str, dict[str, str]]:

    lesson_id = str(uuid.uuid4())
    temp_path = f"{TEMP_DIR}/{course_id}/{lesson_id}.pdf"
    with open(temp_path, "wb") as f:
        content = await pdf_file.read()
        f.write(content)

    config = {"configurable": {"thread_id": lesson_id, "lesson_id": lesson_id}}
    input_state = {
        "type_request": type_request,
        "task": None,
        "result": None,
        "lecture": None,
        "quiz": None,
        "document_path": temp_path,
    }

    response = await graph.ainvoke(input=input_state, config=config)

    lecture = response.get("lecture")
    return lesson_id ,lecture







