import os
import re
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT")
OUTPUT_DIR = ROOT / "Output" / "indobert_v2_kaggle"

def update_documentation():
    summary_csv = OUTPUT_DIR / "exp_indobert_v2_suite_summary.csv"
    if not summary_csv.exists():
        print(f"Summary file {summary_csv} does not exist yet!")
        return False
        
    df = pd.read_csv(summary_csv)
    print("Loaded summary data:")
    print(df)
    
    # We will format this into a markdown table
    # Format: | Varian / Skenario | Strategi Penyeimbangan | Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Recall Netral (Mean) | Status |
    md_rows = []
    for _, row in df.iterrows():
        part = row['part']
        scen = row['scenario']
        strat = row['strategy']
        acc_m = row['accuracy_mean'] * 100
        acc_s = row['accuracy_std'] * 100 if pd.notna(row['accuracy_std']) else 0.0
        f1_m = row['macro_f1_mean'] * 100
        f1_s = row['macro_f1_std'] * 100 if pd.notna(row['macro_f1_std']) else 0.0
        rec_net = row['recall_netral_mean'] * 100
        
        lbl = f"{part.title()} ({scen})" if part == "simulasi" else f"Empiris ({strat.replace('_', ' ').title()})"
        strat_lbl = strat.replace('_', ' ').title()
        
        md_rows.append(
            f"| **{lbl}** | {strat_lbl} | **{acc_m:.2f}%** ± {acc_s:.2f}% | **{f1_m:.2f}%** ± {f1_s:.2f}% | **{rec_net:.2f}%** | **Kaggle GPU T4** (3 Seed) |"
        )
        
    table_str = "\n".join(md_rows)
    print("\nFormatted Table Rows:\n" + table_str)
    
    # Write to temp markdown file for review
    with open(ROOT / "scratch" / "indobert_v2_table.md", "w", encoding="utf-8") as f:
        f.write(table_str)
        
    print("\n[+] Formatted table written to scratch/indobert_v2_table.md")
    return True

if __name__ == "__main__":
    update_documentation()
