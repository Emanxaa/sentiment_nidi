import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT")
DOC_PATH = ROOT / "docs" / "RANGKUMAN_HASIL_EKSPERIMEN_DATA_V2.md"
SUMMARY_PATH = ROOT / "Output" / "predictions" / "indobert_v2_suite_summary.csv"

df_indobert = pd.read_csv(SUMMARY_PATH)

# Let's inspect the exact lines to replace in RANGKUMAN_HASIL_EKSPERIMEN_DATA_V2.md
content = DOC_PATH.read_text(encoding="utf-8")

# Prepare the new markdown section for IndoBERTweet-LoRA on Data Baru V2
new_section = """## 4. IndoBERTweet-LoRA pada Data Baru V2

**Notebook Referensi**: [`notebooks/04_model_indobertweet_lora_data_baru_v2_empiris.ipynb`](../notebooks/04_model_indobertweet_lora_data_baru_v2_empiris.ipynb), [`_simulasi.ipynb`](../notebooks/04_model_indobertweet_lora_data_baru_v2_simulasi.ipynb), & [`notebooks/exp_indobert_v2.ipynb`](../notebooks/exp_indobert_v2.ipynb)  
**Kaggle Kernel Resmi**:
* `emanuelembuaijdak/thesis-indobert-v2` (**GPU Tesla T4**, 27 Runs Suite, Status: **COMPLETE**)
* `emanuelembuaijdak/thesis-lora-p1-fine-tuning-sweep` (GPU Tesla T4, Status: COMPLETE)
* `emanuelembuaijdak/thesis-lora-p2-task-adaptive-pretraining` (GPU Tesla T4, Status: COMPLETE)

### A. Varian Empiris Data Baru V2 (Distribusi Asli)

| Varian Model | Sumber Eksekusi | Test Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Recall Netral (Mean) | F1 Netral (Mean) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Natural)** | **Kaggle GPU T4** (3 Seed) | **81,17%** ± 0,81% | **76,18%** ± 0,94% | 55,08% | 60,38% | Optimal Natural |
| **Class Weight** | **Kaggle GPU T4** (3 Seed) | 77,61% ± 1,66% | 74,08% ± 1,42% | **66,67%** | 59,53% | Prioritas Netral |
| **Random Over-Sampling (ROS)** | **Kaggle GPU T4** (3 Seed) | 77,80% ± 0,31% | 73,89% ± 0,36% | 63,13% | 58,50% | Stabil |
| **Random Under-Sampling (RUS)** | **Kaggle GPU T4** (3 Seed) | 75,01% ± 0,81% | 72,00% ± 0,66% | **70,97%** | 58,37% | Max Recall Netral |
| **P1 Sweep Winner** | Kaggle GPU T4 (Single Run) | 77,98% | 73,90% | 64,10% | 60,12% | Tuning Sweep |
| **P2 TAPT + LoRA (Domain Adapt)**| **Kaggle GPU T4** (Single Run) | **80,06%** | **74,61%** | **68,50%** | **61,20%** | **Adaptasi Bencana** |

### B. Simulasi Ketimpangan Kelas Data Baru V2 (3 Skenario Rasio)

| Skenario Simulasi | Strategi | Sumber Eksekusi | Test Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Recall Netral (Mean) | Status / Ketahanan |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Skenario 1:1:1** (Seimbang) | Baseline | **Kaggle GPU T4** (3 Seed) | **74,53%** ± 1,14% | **71,17%** ± 0,91% | **67,99%** | Unggul +13,01 pp vs BiLSTM |
| **Skenario 1:1:1** (Seimbang) | ROS | **Kaggle GPU T4** (3 Seed) | **74,60%** ± 1,74% | **71,45%** ± 1,46% | 65,56% | Sangat Stabil |
| **Skenario 6:3:1** (Moderat) | Baseline | **Kaggle GPU T4** (3 Seed) | **80,60%** ± 0,24% | **73,45%** ± 0,45% | 40,29% | Unggul +16,29 pp vs BiLSTM |
| **Skenario 6:3:1** (Moderat) | ROS | **Kaggle GPU T4** (3 Seed) | **77,73%** ± 0,51% | **72,92%** ± 0,34% | **55,52%** | Recall Netral Terjaga |
| **Skenario 8:1:1** (Ekstrem) | Baseline | **Kaggle GPU T4** (3 Seed) | **78,52%** ± 0,51% | **70,20%** ± 1,09% | 36,86% | **Kekebalan Mutlak (Anti-Collapse)** |
| **Skenario 8:1:1** (Ekstrem) | ROS | **Kaggle GPU T4** (3 Seed) | **75,57%** ± 2,02% | **71,19%** ± 0,69% | **63,24%** | **Unggul +12,68 pp vs BiLSTM ROS** |
"""

# Replace Section 4 in content
pattern = r"## 4\. IndoBERTweet-LoRA pada Data Baru V2.*?(?=## 5\. Kesimpulan Kritis)"
import re
new_content = re.sub(pattern, new_section + "\n", content, flags=re.DOTALL)

DOC_PATH.write_text(new_content, encoding="utf-8")
print("[+] Successfully updated docs/RANGKUMAN_HASIL_EKSPERIMEN_DATA_V2.md")
