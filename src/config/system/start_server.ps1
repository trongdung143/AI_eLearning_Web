Start-Process -WindowStyle Hidden -WorkingDirectory "E:\Project\chatbot" `
-FilePath "powershell.exe" `
-ArgumentList "-WindowStyle Hidden", "-Command", "& { . .\.venv\Scripts\Activate.ps1; uvicorn src.main:app --host 0.0.0.0 --port 8080 }"
