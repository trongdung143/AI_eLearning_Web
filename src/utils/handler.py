from fastapi import UploadFile, HTTPException
import os

FILE_EXTENSIONS = {
    "documents": [
        ".pdf",
        ".txt",
        ".doc",
        ".docx",
        ".odt",
        ".rtf",
        ".md",
        ".csv",
        ".json",
    ],
    "images": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".tiff"],
    "archives": [".zip", ".rar", ".7z", ".gz"],
}


def get_file_category_and_extension(filename: str) -> tuple[str, str]:
    extension = os.path.splitext(filename)[1].lower()
    for category, extensions in FILE_EXTENSIONS.items():
        if extension in extensions:
            return category, extension
    return "others", extension


def save_upload_file_into_temp(upload_file: UploadFile, conversion_id: str) -> str:
    _, extension = get_file_category_and_extension(upload_file.filename)

    if not extension:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    base_dir = "src/data/temp"
    save_dir = os.path.join(base_dir, conversion_id)
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, upload_file.filename)

    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "wb") as out_file:
        for chunk in iter(lambda: upload_file.file.read(1024 * 1024), b""):
            out_file.write(chunk)

    return upload_file.filename
