from fastapi import APIRouter, Cookie, Response
from fastapi.responses import RedirectResponse, HTMLResponse
from typing import Optional
import uuid

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def get_home(session_id: Optional[str] = Cookie(None)):
    with open("src/static/chat/index.html", "r", encoding="utf-8") as f:
        content = f.read()

    if not session_id:
        new_session_id = str(uuid.uuid4())

        response = HTMLResponse(content=content)
        response.set_cookie(
            key="session_id",
            value=new_session_id,
            httponly=True,
            max_age=60 * 60 * 24 * 7,
        )
        return response

    return HTMLResponse(content=content)


@router.get("/home", response_class=HTMLResponse)
async def get_home_with_session(session_id: Optional[str] = Cookie(None)):
    if not session_id:
        return RedirectResponse(url="/", status_code=302)

    with open("src/static/chat/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@router.get("/logout", response_class=HTMLResponse)
async def get_home_with_session(session_id: Optional[str] = Cookie(None)):
    response = RedirectResponse(url="/", status_code=302)

    if session_id:
        response.delete_cookie("session_id")

    return response
