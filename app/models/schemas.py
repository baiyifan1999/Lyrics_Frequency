from typing import Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    artist: str = Field(..., min_length=1, description="歌手名称")
    genius_token: str = Field(..., min_length=1, description="Genius API Access Token")
    max_songs: int = Field(default=20, ge=1, le=100, description="最多抓取歌曲数量")
    pos_tags: list[str] = Field(
        default=["NOUN", "VERB", "ADJ"],
        description="要统计的词性标签，可选: NOUN VERB ADJ ADV PROPN",
    )
    custom_stop_words: list[str] = Field(
        default=[],
        description="额外停用词列表",
    )
    top_n: int = Field(default=50, ge=1, le=500, description="返回频率最高的前 N 个词")

    model_config = {
        "json_schema_extra": {
            "example": {
                "artist": "Taylor Swift",
                "genius_token": "YOUR_GENIUS_TOKEN",
                "max_songs": 20,
                "pos_tags": ["NOUN", "VERB", "ADJ"],
                "custom_stop_words": [],
                "top_n": 50,
            }
        }
    }


class WordFrequency(BaseModel):
    word: str
    count: int
    pos: str


class POSSummary(BaseModel):
    pos: str
    total_count: int
    top_words: list[WordFrequency]


class AnalyzeResponse(BaseModel):
    artist: str
    songs_analyzed: int
    total_words: int
    pos_summary: list[POSSummary]
    all_top_words: list[WordFrequency]
