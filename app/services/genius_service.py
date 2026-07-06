import re
import lyricsgenius
from functools import lru_cache

VERSION_HINTS = [
    "remaster", "remastered", "live", "demo", "acoustic", "radio edit",
    "edit", "version", "mix", "mono", "stereo", "re-record",
    "deluxe", "anniversary", "extended", "instrumental", "session", "bbc",
]


def get_genius_client(token: str) -> lyricsgenius.Genius:
    return lyricsgenius.Genius(
        token,
        verbose=False,
        skip_non_songs=True,
        remove_section_headers=True,
        timeout=10,
    )


def normalize_title(title: str) -> str:
    t = title.lower().strip()
    for pattern in [r"\([^)]*\)", r"\[[^\]]*\]"]:
        while True:
            m = re.search(pattern, t)
            if not m:
                break
            chunk = m.group(0)[1:-1]
            if any(h in chunk for h in VERSION_HINTS):
                t = t[: m.start()] + " " + t[m.end() :]
            else:
                break
    return re.sub(r"\s+", " ", t).strip()


def clean_lyrics(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r".*?Matches.*?\n", "", raw)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"You might also like.*", " ", text, flags=re.I | re.S)
    text = re.sub(r"\d+Embed", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_lyrics(token: str, artist_name: str, max_songs: int) -> tuple[str, list[str]]:
    """返回 (artist_display_name, [cleaned_lyrics, ...])"""
    genius = get_genius_client(token)
    artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
    if not artist:
        raise ValueError(f"未找到歌手: {artist_name}")

    seen: dict[str, str] = {}
    for song in artist.songs:
        norm = normalize_title(song.title)
        cleaned = clean_lyrics(song.lyrics or "")
        if norm not in seen or len(cleaned) > len(seen[norm]):
            seen[norm] = cleaned

    return artist.name, list(seen.values())
