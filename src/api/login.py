# from fastapi import APIRouter, Request, Response, Depends
# from fastapi.responses import RedirectResponse, JSONResponse
# import requests, uuid
# from src.config.setup import REDIRECT_URI, GOOGLE_CLIENT_SECRET, GOOGLE_CLIENT_ID
# from sqlalchemy.orm import Session

# router = APIRouter()


# @router.get("/login/google")
# async def login_google():
#     state = str(uuid.uuid4())
#     auth_url = (
#         "https://accounts.google.com/o/oauth2/v2/auth"
#         f"?client_id={GOOGLE_CLIENT_ID}"
#         f"&redirect_uri={REDIRECT_URI}"
#         f"&response_type=code"
#         f"&scope=openid%20email%20profile"
#         f"&state={state}"
#         f"&access_type=offline"
#         f"&prompt=consent"
#     )
#     return RedirectResponse(auth_url)


# @router.get("/login/google/callback")
# async def login_google_callback(
#     request: Request,
#     code: str,
#     state: str,
#     response: Response,
#     db: Session = Depends(get_db),
# ):
#     token_url = "https://oauth2.googleapis.com/token"
#     data = {
#         "code": code,
#         "client_id": GOOGLE_CLIENT_ID,
#         "client_secret": GOOGLE_CLIENT_SECRET,
#         "redirect_uri": REDIRECT_URI,
#         "grant_type": "authorization_code",
#     }
#     r = requests.post(token_url, data=data)
#     token_data = r.json()

#     if "access_token" not in token_data:
#         return JSONResponse(
#             {"error": "Can't get token", "detail": token_data}, status_code=400
#         )

#     access_token = token_data["access_token"]
#     refresh_token = token_data.get("refresh_token", "")
#     user_info = requests.get(
#         "https://www.googleapis.com/oauth2/v1/userinfo",
#         params={"access_token": access_token},
#     ).json()
#     email = user_info.get("email")
#     session_id = str(uuid.uuid4())
#     response.set_cookie(
#         key="session_id", value=session_id, httponly=True, max_age=60 * 60 * 24 * 7
#     )
#     try:
#         if email:
#             create_user(
#                 db,
#                 session_id=session_id,
#                 email=email,
#                 access_token=access_token,
#                 refresh_token=refresh_token,
#             )
#     except Exception as e:
#         print(e)

#     response = RedirectResponse(url="/")

#     return response
