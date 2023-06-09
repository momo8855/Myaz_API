from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import user, file, task, youtube
from .config import settings

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user.router)
app.include_router(file.router)
app.include_router(task.router)
app.include_router(youtube.router)