from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Lyrics Frequency API",
    description="输入歌手名字，获取歌词并进行词性频率分析",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", summary="健康检查")
async def root():
    return {"status": "ok", "message": "Lyrics Frequency API is running"}
