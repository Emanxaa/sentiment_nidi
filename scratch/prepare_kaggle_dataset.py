import os
import shutil
import json
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT")
STAGING = ROOT / "staging_kaggle_data"
STAGING.mkdir(parents=True, exist_ok=True)

# 1. Dataset metadata
metadata = {
    "title": "Thesis IndoBERT Processed Data",
    "id": "emanuelembuaijdak/thesis-indobert-processed-data",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(STAGING / "dataset-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# 2. Collect files
files_to_copy = [
    # Data Lama
    ROOT / "Data" / "data_banjir.csv",
    ROOT / "Data" / "data_preprocessed_with_emoticon.csv",
    ROOT / "Data" / "split_data.pkl",
    # Data Baru v2
    ROOT / "Data" / "processed" / "banjir_processed_v2.csv",
    ROOT / "Data" / "raw" / "banjir.csv",
    # Kamus
    ROOT / "kamus" / "colloquial-indonesian-lexicon.csv",
    # Simulasi
    ROOT / "Data" / "simulated" / "scenario_111.csv",
    ROOT / "Data" / "simulated" / "scenario_631.csv",
    ROOT / "Data" / "simulated" / "scenario_811.csv",
    # Predictions
    ROOT / "Output" / "predictions" / "indobert_p1_sweep_test_predictions.csv",
    ROOT / "Output" / "predictions" / "indobert_p2_tapt_test_predictions.csv",
    ROOT / "Output" / "predictions" / "indobert_b03_predictions.csv",
    ROOT / "Output" / "predictions" / "lstm_kaggle_metrics.csv",
    ROOT / "Output" / "predictions" / "bilstm_kaggle_metrics.csv",
]

for src in files_to_copy:
    if src.exists():
        dst = STAGING / src.name
        print(f"Copying {src.name}...")
        shutil.copy2(src, dst)
    else:
        print(f"Warning: {src} not found!")

# Generate data_preprocessed_with_emoticon.pkl for legacy notebook compatibility
pkl_target = STAGING / "data_preprocessed_with_emoticon.pkl"
if not pkl_target.exists():
    print("Generating data_preprocessed_with_emoticon.pkl...")
    df_old = pd.read_csv(ROOT / "Data" / "data_preprocessed_with_emoticon.csv")
    df_old.to_pickle(pkl_target)
    print(f"Generated {pkl_target.name} ({pkl_target.stat().st_size} bytes)")

print("\nFiles in staging directory:")
for f in STAGING.iterdir():
    print(f" - {f.name} ({f.stat().st_size:,} bytes)")
