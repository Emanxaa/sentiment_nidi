"""Konfigurasi terpusat untuk pipeline kualitas data (PRD_DATA_QUALITY_PIPELINE.md)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
REPORTS_DIR = ROOT / "reports"
ANNOTATION_DIR = ROOT / "Annotation"
BATCHES_DIR = ANNOTATION_DIR / "batches"
RESULTS_DIR = ANNOTATION_DIR / "results"
PROMPTS_DIR = ANNOTATION_DIR / "prompts"
ANN_REPORTS_DIR = ANNOTATION_DIR / "reports"

RAW_CSV = DATA_DIR / "data_banjir.csv"
PREPROCESSED_CSV = DATA_DIR / "data_preprocessed_with_emoticon.csv"
V2_CSV = DATA_DIR / "data_banjir_v2.csv"
V2_PREPROCESSED_CSV = DATA_DIR / "data_preprocessed_with_emoticon_v2.csv"
SPLIT_PKL = DATA_DIR / "split_data.pkl"
SPLIT_V2_PKL = DATA_DIR / "split_data_v2.pkl"

# Sampling & batching (Phase 2 & 3)
N_SAMPLES = 1000
BATCH_SIZE = 50
N_BATCHES = N_SAMPLES // BATCH_SIZE  # 20
RANDOM_STATE = 42

# Label
LABEL_MAP = {"negatif": 0, "netral": 1, "positif": 2}
LABEL_NAMES = {0: "negatif", 1: "netral", 2: "positif"}

# Phase 3 (LLM)
LLM_MODEL = "gpt-4o-mini"
CONFIDENCE_THRESHOLD = 80

# Phase 1 — audit
CRITICAL_STOPWORDS = ["tidak", "bukan", "jangan", "sedih", "marah", "doa", "harapan"]
NEGATION_WORDS = ["tidak", "bukan", "jangan", "belum"]

# --- Pola noise (Task 0.3) ---
NOISE_PATTERNS = {
    "tampilkan_lebih_banyak": re.compile(r"tampilkan\s+lebih\s+banyak", re.IGNORECASE),
    "engagement_rb": re.compile(r"\b\d+\s*rb\b", re.IGNORECASE),
    "timestamp_video": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    "sumber_berita": re.compile(r"\bdari\s+[\w.\-]+\.(com|id|co|net)\b", re.IGNORECASE),
    "noise_token": re.compile(r"\b(?:rb|com|id|co|net|membalas)\b", re.IGNORECASE),
}

# --- Cleaning rules baru (Task 0.4) ---
# Setiap aturan: (nama, regex, pengganti). Diterapkan berurutan, lalu spasi dirapikan.
CLEANING_RULES = [
    ("tampilkan_lebih_banyak", re.compile(r"tampilkan\s+lebih\s+banyak", re.IGNORECASE), " "),
    ("engagement_rb", re.compile(r"\b\d+\s*rb\b", re.IGNORECASE), " "),
    ("timestamp_video", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"), " "),
    ("sumber_berita", re.compile(r"\bdari\s+[\w.\-]+\.(com|id|co|net)\b", re.IGNORECASE), " "),
    ("noise_token", re.compile(r"\b(com|id|co|net|rb|membalas)\b", re.IGNORECASE), " "),
]
