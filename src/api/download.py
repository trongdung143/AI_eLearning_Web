from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()


@router.get("/download/{file_name}")
async def get_file(file_name: str):
    folder_path = f"src/data/{file_name}"
    if not os.path.exists(folder_path):
        return {"error": "File not found"}

    return FileResponse(
        path=folder_path, filename=file_name, media_type="application/octet-stream"
    )
