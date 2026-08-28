"""Phase 2 — Gold Dataset Creation (PRD).

Stratified Random Sampling 1000 tweet, random_state=42, menambahkan kolom id
(= posisi baris asli di data_preprocessed_with_emoticon.csv, stabil untuk mapping balik).

Output: Annotation/gold_dataset_1000.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from quality_pipeline import config as C
from quality_pipeline.utils import load_csv, write_csv

OUTPUT_COLUMNS = [
    "id", "text", "text_with_emoticon", "sentimen", "label",
    "created_at", "keyword",
]


def run(dry_run: bool = False) -> None:
    print("Phase 2 — Gold Dataset Creation")
    df = load_csv(C.PREPROCESSED_CSV)
    print(f"  Memuat {len(df)} baris; distribusi sentimen:\n{df['sentimen'].value_counts().to_string()}")

    n = C.N_SAMPLES
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n / len(df), random_state=C.RANDOM_STATE)
    for _train_idx, test_idx in sss.split(df, df["sentimen"]):
        sample = df.iloc[test_idx].copy()

    sample["id"] = sample.index  # posisi baris asli
    out = sample[OUTPUT_COLUMNS].reset_index(drop=True)
    write_csv(out, C.ANNOTATION_DIR / "gold_dataset_1000.csv")

    print(f"  Sampel: {len(out)} baris -> Annotation/gold_dataset_1000.csv")
    print("  Distribusi sampel:")
    print(out["sentimen"].value_counts().to_string())
    assert len(out) == n, f"Jumlah sampel {len(out)} != {n}"
    assert out["id"].is_unique, "Kolom id harus unik"
    print("  OK: jumlah 1000, id unik.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 — Gold Dataset Creation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
