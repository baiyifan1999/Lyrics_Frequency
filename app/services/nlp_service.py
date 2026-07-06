from collections import Counter
import spacy

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def analyze_lyrics(
    lyrics_list: list[str],
    allowed_pos: set[str],
    stop_words: set[str],
) -> tuple[Counter, Counter]:
    """
    返回:
      word_counter  — {word: count}
      pos_counter   — {(word, pos): count}
    """
    nlp = get_nlp()
    word_counter: Counter = Counter()
    pos_counter: Counter = Counter()

    for text in lyrics_list:
        if not text:
            continue
        doc = nlp(text)
        for token in doc:
            word = token.text.lower()
            if (
                token.pos_ in allowed_pos
                and word.isalpha()
                and word not in stop_words
                and len(word) > 1
            ):
                word_counter[word] += 1
                pos_counter[(word, token.pos_)] += 1

    return word_counter, pos_counter


def build_stop_words(extra: list[str]) -> set[str]:
    nlp = get_nlp()
    stops = nlp.Defaults.stop_words.copy()
    stops.update(w.strip().lower() for w in extra if w.strip())
    return stops
