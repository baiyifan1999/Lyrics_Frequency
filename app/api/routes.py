from collections import Counter

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    POSSummary,
    WordFrequency,
)
from app.services.genius_service import fetch_lyrics
from app.services.nlp_service import analyze_lyrics, build_stop_words

router = APIRouter()

VALID_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


@router.post("/analyze", response_model=AnalyzeResponse, summary="歌词词性频率分析")
async def analyze(req: AnalyzeRequest):
    invalid = set(req.pos_tags) - VALID_POS
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"不支持的词性标签: {invalid}。可选: {VALID_POS}",
        )

    try:
        artist_name, lyrics_list = fetch_lyrics(
            req.genius_token, req.artist, req.max_songs
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Genius API 错误: {e}")

    stop_words = build_stop_words(req.custom_stop_words)
    word_counter, pos_counter = analyze_lyrics(
        lyrics_list, set(req.pos_tags), stop_words
    )

    if not word_counter:
        raise HTTPException(status_code=422, detail="未能从歌词中提取到有效词语")

    # 按词性汇总
    pos_buckets: dict[str, Counter] = {p: Counter() for p in req.pos_tags}
    for (word, pos), count in pos_counter.items():
        if pos in pos_buckets:
            pos_buckets[pos][word] += count

    pos_summary = [
        POSSummary(
            pos=pos,
            total_count=sum(c.values()),
            top_words=[
                WordFrequency(word=w, count=c, pos=pos)
                for w, c in bucket.most_common(req.top_n)
            ],
        )
        for pos, bucket in pos_buckets.items()
        if (bucket := pos_buckets[pos])
    ]

    all_top = [
        WordFrequency(word=w, count=c, pos="MIXED")
        for w, c in word_counter.most_common(req.top_n)
    ]

    return AnalyzeResponse(
        artist=artist_name,
        songs_analyzed=len([l for l in lyrics_list if l]),
        total_words=sum(word_counter.values()),
        pos_summary=pos_summary,
        all_top_words=all_top,
    )
