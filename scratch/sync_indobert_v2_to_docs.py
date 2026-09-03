import os
import json
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT")
OUTPUT_DIR = ROOT / "Output" / "indobert_v2_kaggle"

def sync_results():
    suite_summary_file = OUTPUT_DIR / "exp_indobert_v2_suite_summary.csv"
    suite_results_file = OUTPUT_DIR / "exp_indobert_v2_suite_results.csv"
    
    if not suite_summary_file.exists():
        print(f"File {suite_summary_file} does not exist yet. Make sure to download kaggle output first.")
        return False
        
    df_summary = pd.read_csv(suite_summary_file)
    df_results = pd.read_csv(suite_results_file)
    
    print("\n" + "="*80)
    print("INDOBERTWEET-LORA DATA BARU V2 - SUITE SUMMARY (3 SEEDS)")
    print("="*80)
    print(df_summary.to_string(index=False))
    
    # Save a copy to Output/predictions for permanent provenance
    pred_dir = ROOT / "Output" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(pred_dir / "indobert_v2_suite_summary.csv", index=False)
    df_results.to_csv(pred_dir / "indobert_v2_suite_results.csv", index=False)
    print(f"\n[+] Saved copy to {pred_dir / 'indobert_v2_suite_summary.csv'}")
    
    return True

if __name__ == "__main__":
    sync_results()
