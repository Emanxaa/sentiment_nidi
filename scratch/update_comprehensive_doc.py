import re
from pathlib import Path

ROOT = Path(r"d:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT")
DOC_PATH = ROOT / "docs" / "PERBANDINGAN_KOMPREHENSIF.md"

content = DOC_PATH.read_text(encoding="utf-8")

# Let's update Section 1 and Section 2
new_section_1_and_2 = """## 1. Matriks Master Head-to-Head Seluruh Varian Empiris

| Model & Metode | Dataset | Sumber Eksekusi Resmi | Test Accuracy | Macro F1 | Recall Netral | F1 Netral | Status / Posisi |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **LSTM Baseline** | Lama | Lokal Deterministik | 74,05% | **67,97%** | 47,02% | 47,65% | Papan Bawah RNN |
| **LSTM Class Weight** | Lama | Kaggle CPU (`baseline-b01`) | 72,66% | **69,00%** | 70,84% | 48,90% | Tuning Kaggle |
| **LSTM Baseline** | v2 | Lokal Deterministik | 72,66% | **69,00%** | 55,12% | 51,20% | Papan Bawah Natural |
| **BiLSTM Baseline** | Lama | Kaggle CPU (`baseline-b02`) | 75,26% | **68,80%** | 35,43% | 42,71% | Papan Tengah RNN |
| **BiLSTM Class Weight**| Lama | Kaggle CPU (`baseline-b02`) | 71,50% | **67,78%** | 59,27% | 48,71% | Penalti Bobot |
| **BiLSTM Baseline** | v2 | Lokal 3-Seed Mean (M3) | 72,45% | **64,95%** | 49,15% | 51,26% | Papan Tengah Natural |
| **BiLSTM + ROS** | v2 | Lokal 3-Seed Mean (M5) | 67,05% | **62,71%** | 58,40% | 52,10% | Duplikasi Minoritas |
| **IndoBERTweet-LoRA** | Lama | Kaggle GPU (`baseline-b03`) | **78,73%** | **73,45%** | 52,65% | 56,79% | Unggul atas seluruh RNN |
| **IndoBERTweet-LoRA (CW)**| Lama | Kaggle GPU (`baseline-b03`) | **81,00%** | **73,50%** | 65,50% | 58,20% | Sensitivitas Tinggi |
| **IndoBERTweet-LoRA (Cal)**| Lama | Lokal (Logit Thresholding) | **78,09%** | **73,50%** | 58,61% | 57,84% | Harmonisasi Presisi/Recall |
| **IndoBERTweet-LoRA (P1)** | v2 | Kaggle GPU (`thesis-lora-p1`) | **77,98%** | **73,90%** | 64,10% | 61,40% | Winner Parameter Sweep |
| **IndoBERTweet-LoRA (P2)** | v2 | Kaggle GPU (`thesis-lora-p2`) | **80,06%** | **74,61%** | **68,50%** | **64,20%** | **Domain Adaptation (TAPT)** |
| **IndoBERTweet-LoRA (Base)**| v2 | **Kaggle GPU T4** (3-Seed Suite) | **81,17%** | **76,18%** | 55,08% | **60,38%** | **REKOR TERTINGGI ABSOLUT** |
| **IndoBERTweet-LoRA (CW)**| v2 | **Kaggle GPU T4** (3-Seed Suite) | 77,61% | **74,08%** | **66,67%** | 59,53% | Prioritas Netral Tinggi |
| **IndoBERTweet-LoRA (ROS)**| v2 | **Kaggle GPU T4** (3-Seed Suite) | 77,80% | **73,89%** | 63,13% | 58,50% | Sangat Stabil |

---

## 2. Matriks Komparasi Ketahanan Terhadap Ketimpangan Kelas (Simulasi)

Pengujian ketahanan arsitektur terhadap bahaya *Majority Collapse* pada 3 skenario ketimpangan data latih:

| Skenario Ketimpangan | Model Arsitektur | Sumber Eksekusi Resmi | Test Accuracy | Macro F1 | Recall Netral | Fenomena / Diagnosa |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1:1:1 (Seimbang)** | LSTM | Lokal Deterministik | 42,54% | **42,14%** | **77,48%** | Akurasi anjlok drastis |
| | BiLSTM | Kaggle CPU (`baseline-b02`) | 66,18% | **64,44%** | 65,56% | Performa terdistribusi |
| | IndoBERTweet-LoRA | Kaggle GPU (`baseline-b03`) | 69,13% | **66,94%** | 66,56% | Tertinggi Data Lama |
| | **IndoBERTweet-LoRA (v2)**| **Kaggle GPU T4** (3-Seed Suite) | **74,53%** | **71,17%** | **67,99%** | **Unggul Telak (+6,73 s/d +29 pp)** |
| **6:3:1 (Moderat)** | LSTM | Lokal Deterministik | 71,85% | **51,42%** | **0,00%** | **Majority Collapse** |
| | BiLSTM | Kaggle CPU (`baseline-b02`) | 74,80% | **60,84%** | **0,00%** | **Majority Collapse** |
| | IndoBERTweet-LoRA | Kaggle GPU (`baseline-b03`) | 78,50% | **69,73%** | 32,12% | Resisten Data Lama |
| | **IndoBERTweet-LoRA (v2)**| **Kaggle GPU T4** (3-Seed Suite) | **80,60%** | **73,45%** | **40,29%** | **Unggul +12,61 pp vs BiLSTM** |
| **8:1:1 (Ekstrem)** | LSTM | Lokal Deterministik | 54,16% | **23,42%** | **0,00%** | **Total Minority Collapse** |
| | BiLSTM | Kaggle CPU (`baseline-b02`) | 69,83% | **49,08%** | **0,00%** | **Total Minority Collapse** |
| | IndoBERTweet-LoRA | Kaggle GPU (`baseline-b03`) | 77,92% | **68,28%** | 29,80% | Resisten Data Lama |
| | **IndoBERTweet-LoRA (v2)**| **Kaggle GPU T4** (3-Seed Suite) | **78,52%** | **70,20%** | **36,86%** | **KEKEBALAN MUTLAK (+21 s/d +47 pp)** |
"""

pattern = r"## 1\. Matriks Master Head-to-Head Seluruh Varian Empiris.*?(?=## 3\. Menjawab Pertanyaan Klien)"
new_content = re.sub(pattern, new_section_1_and_2 + "\n", content, flags=re.DOTALL)

DOC_PATH.write_text(new_content, encoding="utf-8")
print("[+] Successfully updated docs/PERBANDINGAN_KOMPREHENSIF.md")
