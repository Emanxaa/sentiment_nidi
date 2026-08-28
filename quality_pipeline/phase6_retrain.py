"""Phase 6 — Retraining Harness (PRD).

Membangun ulang artefak pelatihan dari data_banjir_v2.csv:
- Data/data_preprocessed_with_emoticon_v2.csv
- Data/split_data_v2.pkl  (kunci sama persis dgn split_data.pkl lama, 12 kunci)

--activate-v2 : backup data_banjir.csv -> data_banjir_v1_backup.csv, lalu salin
                v2 + artefaknya ke nama kanonik agar notebook 01-04 bisa langsung
                membaca. DESTRUKTIF (hanya dengan flag eksplisit).
"""
from __future__ import annotations

import argparse
import pickle
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from quality_pipeline import config as C
from quality_pipeline.preprocess import add_preprocessed_columns, build_text_with_emoticon
from quality_pipeline.utils import load_csv, write_csv

SPLIT_KEYS = [
    "df_train", "df_test", "X_train_lstm", "X_test_lstm",
    "X_train_bert", "X_test_bert", "y_train", "y_test",
    "X_training_lstm", "X_validation_lstm", "y_training_lstm", "y_validation_lstm",
]

CANONICAL_COLUMNS = [
    "text", "clean_text", "created_at", "keyword", "processed_text",
    "sentimen", "label", "emoticon", "text_with_emoticon_raw",
    "text_with_emoticon", "clean_text_lstm", "text_bert",
]


def build_v2_artifacts(source_csv=C.V2_CSV) -> pd.DataFrame:
    if not source_csv.exists():
        raise SystemExit(
            f"{source_csv.name} belum ada. Jalankan Phase 2-4 dulu untuk membentuk v2."
        )
    print(f"  Memuat {source_csv.name}...")
    df = load_csv(source_csv)

    print("  Membangun text_with_emoticon, clean_text_lstm, text_bert...")
    df = build_text_with_emoticon(df)
    df = add_preprocessed_columns(
        df,
        on_progress=lambda n: print(f"    +{n} baris selesai", flush=True),
    )
    df["label"] = df["sentimen"].map(C.LABEL_MAP).astype(int)

    cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    df = df[cols]
    write_csv(df, C.V2_PREPROCESSED_CSV)
    print(f"  -> Data/data_preprocessed_with_emoticon_v2.csv ({len(df)} baris)")

    print("  Membagi train/test (80/20, stratify, random_state=42)...")
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=C.RANDOM_STATE
    )
    X_train_lstm = df_train["clean_text_lstm"].reset_index(drop=True)
    X_test_lstm = df_test["clean_text_lstm"].reset_index(drop=True)
    X_train_bert = df_train["text_bert"].reset_index(drop=True)
    X_test_bert = df_test["text_bert"].reset_index(drop=True)
    y_train = df_train["label"].reset_index(drop=True)
    y_test = df_test["label"].reset_index(drop=True)
    X_training_lstm, X_validation_lstm, y_training_lstm, y_validation_lstm = train_test_split(
        X_train_lstm, y_train, test_size=0.1, stratify=y_train, random_state=32
    )

    data = {
        "df_train": df_train,
        "df_test": df_test,
        "X_train_lstm": X_train_lstm,
        "X_test_lstm": X_test_lstm,
        "X_train_bert": X_train_bert,
        "X_test_bert": X_test_bert,
        "y_train": y_train,
        "y_test": y_test,
        "X_training_lstm": X_training_lstm,
        "X_validation_lstm": X_validation_lstm,
        "y_training_lstm": y_training_lstm,
        "y_validation_lstm": y_validation_lstm,
    }
    with open(C.SPLIT_V2_PKL, "wb") as f:
        pickle.dump(data, f)
    print(f"  -> Data/split_data_v2.pkl (train {len(df_train)} / test {len(df_test)} / val {len(X_validation_lstm)})")
    return df


def activate_v2() -> None:
    """Backup data lama lalu pakai v2 sebagai kanonik (untuk notebook lama)."""
    backups = {
        C.RAW_CSV: C.DATA_DIR / "data_banjir_v1_backup.csv",
        C.PREPROCESSED_CSV: C.DATA_DIR / "data_preprocessed_with_emoticon_v1_backup.csv",
        C.SPLIT_PKL: C.DATA_DIR / "split_data_v1_backup.pkl",
    }
    for src, bak in backups.items():
        if src.exists():
            shutil.copy2(src, bak)
            print(f"  backup: {src.name} -> {bak.name}")
    shutil.copy2(C.V2_CSV, C.RAW_CSV)
    shutil.copy2(C.V2_PREPROCESSED_CSV, C.PREPROCESSED_CSV)
    shutil.copy2(C.SPLIT_V2_PKL, C.SPLIT_PKL)
    print("  Aktifkan v2: data_banjir.csv, data_preprocessed_with_emoticon.csv, split_data.pkl sudah memakai v2.")
    print("  Notebook 01-04 kini dapat dijalankan langsung (random_state sama -> hasil komparabel).")


def run(dry_run: bool = False, activate: bool = False) -> None:
    print("Phase 6 — Retraining Harness")
    build_v2_artifacts()
    print("  Distribusi sentimen pada v2:")
    print(load_csv(C.V2_PREPROCESSED_CSV)["sentimen"].value_counts().to_string())

    if activate:
        print("\nMengaktifkan v2 sebagai dataset kanonik...")
        activate_v2()
    else:
        print("\n[MANUAL] Jalankan retrain di Kaggle/Colab (butuh GPU):")
        print("  1. Upload Data/data_preprocessed_with_emoticon_v2.csv & Data/split_data_v2.pkl")
        print("  2. Rename jadi data_preprocessed_with_emoticon.csv & split_data.pkl (atau jalankan phase6 --activate-v2 dulu).")
        print("  3. Jalankan ulang 02_model_lstm.ipynb, 03_model_bilstm.ipynb, 04_model_indobertweet_lora.ipynb.")
        print("  4. Catat Macro F1 tiap model dan bandingkan dengan hasil sebelum v2 (baseline).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6 — Retraining Harness")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--activate-v2", action="store_true", help="Backup & pakai v2 sebagai dataset kanonik")
    args = ap.parse_args()
    run(dry_run=args.dry_run, activate=args.activate_v2)


if __name__ == "__main__":
    main()
