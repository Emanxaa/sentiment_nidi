"""Reproduksi fungsi preprocessing dari 01_preprocessing.ipynb.

Dipakai ulang oleh Phase 1 (audit) dan Phase 6 (retraining). Import memakai
ejaan folder yang benar ("Preprocessing", bukan "Prepocessing" seperti di notebook).

Stemming Sastrawi lambat, jadi Phase 6 menggunakan strategi "stem sekali per token
unik": seluruh token kandidat dikumpulkan, token uniknya di-stem secara paralel,
lalu setiap tweet direkonstruksi lewat peta stem.
"""
from __future__ import annotations

import concurrent.futures
import os
import re

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from Preprocessing.emoji_dict import emoji_dict
from Preprocessing.normalisasi_dict import normalisasi_dict
from Preprocessing.stopwords_lstm_processing import keep_words

_stopwords = set(StopWordRemoverFactory().get_stop_words()) - keep_words
_stemmer = StemmerFactory().create_stemmer()
_stem_cache: dict[str, str] = {}


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _stem_tokens_chunk(tokens):
    """Worker: stem daftar token unik (dipakai ProcessPoolExecutor)."""
    return [_stemmer.stem(w) for w in tokens]


def _default_n_jobs(n_jobs=None):
    if n_jobs is None:
        return max(1, min(os.cpu_count() or 1, 10))
    return n_jobs


def parallel_stem_tokens(unique_tokens, n_jobs=None):
    """Stem setiap token unik tepat satu kali (paralel) -> dict token->stem."""
    n_jobs = _default_n_jobs(n_jobs)
    if n_jobs <= 1 or len(unique_tokens) < 200:
        return {w: _stemmer.stem(w) for w in unique_tokens}
    chunk_size = max(1, len(unique_tokens) // n_jobs)
    chunks = list(_chunks(unique_tokens, chunk_size))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(ex.map(_stem_tokens_chunk, chunks))
    return {w: s for chunk, part in zip(chunks, parts) for w, s in zip(chunk, part)}


def _lstm_tokens(text) -> list[str]:
    """Token kandidat stemming untuk preprocess_lstm (tanpa stemming)."""
    if pd.isna(text):
        return []
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    text = re.sub(r"\bcom\b", "", text)
    text = re.sub(r"\brb\b", "", text)
    text = re.sub(r"\btampilkan\b", "", text)
    text = re.sub(r"\bmembalas\b", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [normalisasi_dict.get(w, w) for w in tokens]
    return [w for w in tokens if w not in _stopwords]


def preprocess_lstm(text, stem_map=None):
    """Preprocessing untuk LSTM/BiLSTM (sama dgn notebook cell 14).

    stem_map: dict token->stem bila tersedia (dari parallel_stem_tokens).
    """
    tokens = _lstm_tokens(text)
    if not tokens:
        return ""
    if stem_map is None:
        result = []
        for word in tokens:
            if word not in _stem_cache:
                _stem_cache[word] = _stemmer.stem(word)
            result.append(_stem_cache[word])
        return " ".join(result)
    return " ".join(stem_map.get(word, _stemmer.stem(word)) for word in tokens)


def preprocess_bert(text):
    """Preprocessing untuk IndoBERTweet-LoRA (sama dgn notebook cell 16)."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert_emoticon(text):
    """Konversi emoji ke kata Indonesia (sama dgn notebook cell 8)."""
    if pd.isna(text):
        return ""
    text = str(text)
    for emo in sorted(emoji_dict.keys(), key=len, reverse=True):
        text = text.replace(emo, emoji_dict[emo])
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text_with_emoticon(df):
    """Bangun text_with_emoticon_raw & text_with_emoticon (sama dgn notebook cell 8)."""
    df = df.copy()
    df["emoticon"] = df["emoticon"].fillna("")
    df["text_with_emoticon_raw"] = df["text"].astype(str) + " " + df["emoticon"].astype(str)
    df["text_with_emoticon"] = df["text_with_emoticon_raw"].apply(convert_emoticon)
    return df


def add_preprocessed_columns(df, n_jobs=None, on_progress=None):
    """Tambahkan clean_text_lstm & text_bert ke DataFrame (sama dgn notebook cell 17).

    Cepat: token unik di-stem sekali secara paralel, lalu tweet direkonstruksi.
    """
    df = df.copy()
    texts = df["text_with_emoticon"].tolist()

    token_lists = [_lstm_tokens(t) for t in texts]
    unique = sorted({w for toks in token_lists for w in toks})
    print(f"  Token unik untuk stemming: {len(unique)}")
    stem_map = parallel_stem_tokens(unique, n_jobs=n_jobs)

    df["clean_text_lstm"] = [
        " ".join(stem_map.get(w, w) for w in toks) for toks in token_lists
    ]
    df["text_bert"] = df["text_with_emoticon"].apply(preprocess_bert)
    if on_progress:
        on_progress(len(texts))
    return df
