"""Utilitas bersama: IO aman, pembersihan teks, deteksi near-duplicate."""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from quality_pipeline.config import CLEANING_RULES

# Karakter Matematical Alphanumeric (U+1D400–U+1D7FF) — artefak hasil scraping.
MATH_ALNUM = frozenset(chr(cp) for cp in range(0x1D400, 0x1D800))


def load_csv(path, **kwargs):
    """Muat CSV dengan fallback encoding (utf-8-sig, utf-8, cp1252, latin-1).

    - cp1252 menangani file yang disimpan ulang oleh Excel Windows.
    - Delimiter dideteksi otomatis (Excel Windows sering memakai ';').
    """
    p = Path(path)
    if "sep" not in kwargs:
        first = p.read_bytes().split(b"\n", 1)[0]
        kwargs["sep"] = ";" if first.count(b";") > first.count(b",") else ","
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Tidak bisa membaca file: {p}")


def write_csv(df, path, **kwargs):
    """Tulis DataFrame ke CSV UTF-8, buat folder induk bila perlu."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8", **kwargs)
    return p


def load_dotenv(path=None):
    """Muat variabel KEY=VALUE dari file .env (tanpa dependensi eksternal).

    Tidak menimpa variabel yang sudah ada di environment.
    """
    if path is None:
        from quality_pipeline import config as C
        path = C.ROOT / ".env"
    p = Path(path)
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def strip_math_letters(text: str) -> str:
    """Hapus huruf matematika tebal/miring (artefak scraping)."""
    return "".join(ch for ch in text if ch not in MATH_ALNUM)


def has_math_letters(text: str) -> bool:
    return any(ch in MATH_ALNUM for ch in safe_str(text))


def apply_cleaning_rules(text: str) -> str:
    """Terapkan CLEANING_RULES (Task 0.4) ke teks lalu rapikan spasi."""
    t = safe_str(text)
    t = strip_math_letters(t)
    for _name, pattern, repl in CLEANING_RULES:
        t = pattern.sub(repl, t)
    return re.sub(r"\s+", " ", t).strip()


def norm_for_dup(text: str) -> str:
    """Normalisasi teks untuk perbandingan near-duplicate."""
    t = unicodedata.normalize("NFKC", safe_str(text)).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def near_duplicate_pairs(norms, threshold=0.9, max_pairs=5000):
    """Cari pasangan near-duplicate via cosine similarity TF-IDF (vektorisasi per blok).

    Mengembalikan list tuple (i, j, sim) dengan i < j.
    """
    if len(norms) < 2:
        return []
    vec = TfidfVectorizer(analyzer=lambda t: t.split(), sublinear_tf=True)
    X = vec.fit_transform(norms)
    n = X.shape[0]
    block = 256
    pairs = []
    for start in range(0, n, block):
        end = min(n, start + block)
        sim = cosine_similarity(X[start:end], X)  # (block, n)
        for r in range(end - start):
            gi = start + r
            row = sim[r, gi + 1:]
            cols = np.where(row >= threshold)[0] + (gi + 1)
            for c in cols:
                pairs.append((gi, int(c), round(float(sim[r, int(c)]), 4)))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs
