import streamlit as st
import lyricsgenius
import spacy
from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==========================================
# 1. 配置与初始化
# ==========================================
st.set_page_config(page_title="歌词词频可视化工具", page_icon="🎵", layout="wide")

VERSION_HINTS = [
    "remaster", "remastered", "live", "demo", "acoustic", "radio edit",
    "edit", "version", "mix", "mono", "stereo", "re-record",
    "deluxe", "anniversary", "extended", "instrumental", "session", "bbc",
]


@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")


@st.cache_resource
def get_genius_client(token):
    return lyricsgenius.Genius(token, verbose=False, skip_non_songs=True, remove_section_headers=True)


# ==========================================
# 2. 数据处理函数
# ==========================================
def normalize_title(title: str) -> str:
    t = title.lower().strip()
    for pattern in [r"\([^)]*\)", r"\[[^\]]*\]"]:
        while True:
            m = re.search(pattern, t)
            if not m: break
            chunk = m.group(0)[1:-1]
            if any(h in chunk for h in VERSION_HINTS):
                t = t[:m.start()] + " " + t[m.end():]
            else:
                break
    return re.sub(r"\s+", " ", t).strip()


def clean_lyrics(raw: str) -> str:
    if not raw: return ""
    text = re.sub(r".*?Matches.*?\n", "", raw)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"You might also like.*", " ", text, flags=re.I | re.S)
    text = re.sub(r"\d+Embed", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def process_text_stream(text, nlp, allowed_pos, stop_words):
    words = []
    doc = nlp(text)
    for token in doc:
        word = token.text.lower()
        if (token.pos_ in allowed_pos and word.isalpha() and word not in stop_words and len(word) > 1):
            words.append(word)
    return words


# ==========================================
# 3. 主界面布局 (输入框已移至此处)
# ==========================================
st.title("🎵 歌词词频分析与可视化")

# 主页输入区域
col_left, col_right = st.columns(2)
with col_left:
    api_token = st.text_input(
        "🔑 Genius Access Token",
        type="password",
        help="在此输入你的 Genius API 密钥"
    )
    st.info("💡 [点击此处跳转 Genius API 管理页面](https://genius.com/api-clients) 获取你的 Token。")

with col_right:
    artist_name = st.text_input("👤 歌手名称", placeholder="例如: Taylor Swift")

# 侧边栏保留高级设置
with st.sidebar:
    st.header("⚙️ 统计设置")
    max_songs = st.slider("➕ 最大歌曲数量", 5, 100, 20)

    pos_options = {
        "名词 (NOUN)": "NOUN", "动词 (VERB)": "VERB", "形容词 (ADJ)": "ADJ",
        "副词 (ADV)": "ADV", "专有名词 (PROPN)": "PROPN"
    }
    selected_pos_friendly = st.multiselect(
        "📝 选择统计词性",
        options=list(pos_options.keys()),
        default=["名词 (NOUN)", "动词 (VERB)", "形容词 (ADJ)"]
    )
    allowed_pos = [pos_options[p] for p in selected_pos_friendly]
    custom_stop_words = st.text_area("🚫 自定义停用词 (逗号隔开)")

# 大按钮置于主页
start_button = st.button("🚀 开始分析", type="primary", use_container_width=True)

# ==========================================
# 4. 执行逻辑
# ==========================================
if start_button:
    if not api_token or not artist_name:
        st.warning("⚠️ 请确保已输入 Token 和 歌手名称。")
        st.stop()

    nlp = load_nlp()
    genius = get_genius_client(api_token)
    stop_words = nlp.Defaults.stop_words.copy()
    if custom_stop_words:
        stop_words.update([s.strip().lower() for s in custom_stop_words.split(",")])

    status_text = st.empty()
    progress_bar = st.progress(0)

    status_text.text(f"正在获取 {artist_name} 的歌曲...")
    try:
        artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
        if not artist:
            st.error("未找到该歌手。")
            st.stop()

        status_text.text(f"已找到 {artist_name}。正在清洗歌词...")  # 这里修正了 args_artist 错误

        best_songs_lyrics = {}
        for s in artist.songs:
            norm = normalize_title(s.title)
            cleaned = clean_lyrics(s.lyrics or "")
            if norm not in best_songs_lyrics or len(cleaned) > len(best_songs_lyrics[norm]):
                best_songs_lyrics[norm] = cleaned

        num_unique_songs = len(best_songs_lyrics)
        all_words = []

        song_lyrics_list = list(best_songs_lyrics.values())
        for i, lyrics_text in enumerate(song_lyrics_list):
            all_words.extend(process_text_stream(lyrics_text, nlp, allowed_pos, stop_words))
            progress_bar.progress((i + 1) / num_unique_songs)
            status_text.text(f"分析进度: {i + 1}/{num_unique_songs}")

        status_text.empty()
        progress_bar.empty()

        if not all_words:
            st.warning("未统计到有效单词。")
            st.stop()

        # 视觉化展示
        word_freq = Counter(all_words)
        df_freq = pd.DataFrame(word_freq.most_common(50), columns=['词语', '出现次数'])

        st.divider()
        st.subheader(f"📊 {artist_name} 歌词词频分析结果")

        c1, c2 = st.columns([1, 1])
        with c1:
            wc = WordCloud(width=800, height=500, background_color="white",
                           colormap="plasma").generate_from_frequencies(word_freq)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        with c2:
            st.bar_chart(df_freq.head(20).set_index('词语'))
            with st.expander("查看完整数据表"):
                st.write(df_freq)

    except Exception as e:
        st.error(f"发生错误: {e}")