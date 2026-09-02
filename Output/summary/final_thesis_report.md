# Laporan Akhir Eksperimen Tesis — Pemodelan LSTM & Strategi Penyeimbangan Data

**Tanggal:** `2026-09-02 21:05:27`  
**Model Arsitektur:** Reusable PyTorch Sentiment LSTM (`Embedding=128`, `LSTM Units=128`, `Dropout=0.3`)  
**Konfigurasi Pelatihan:** Adam (`lr=0.0005`, `batch_size=16`, `patience=3`, `max_epochs=20`)  
**Protokol Evaluasi:** 3 Random Seeds Independen (`42`, `123`, `456`), Zero Data Leakage  

---

## 1. Dataset Overview

* **Sumber Data:** `Data/processed/banjir_processed_v2.csv`
* **Total Sampel:** 8,648 tweets berbahasa Indonesia mengenai banjir.
* **Pembagian Data:**
  * **Train Set:** 6,226 sampel (72% dari total dataset)
  * **Validation Set:** 692 sampel (8% dari total dataset, 10% dari partisi Train)
  * **Test Set:** 1,730 sampel (20% dari total dataset, hold-out murni)
* **Distribusi Kelas Asli:**
  * **Negative (0):** 4,687 sampel (54.19%)
  * **Positive (2):** 2,451 sampel (28.34%)
  * **Neutral (1):** 1,510 sampel (17.46%) — *Minority Class*

---

## 2. Experimental Design (Metode Penyeimbangan)

Eksperimen mengevaluasi 5 strategi penanganan *class imbalance*:
1. **Baseline (M3):** Pelatihan standar tanpa manipulasi bobot atau distribusi sampel.
2. **Class Weight (M4):** Cost-sensitive learning menggunakan `CrossEntropyLoss(weight=class_weights)`.
3. **Random Oversampling / ROS (M5):** Duplikasi sampel minoritas dengan pengembalian hingga berukuran sama dengan kelas mayoritas.
4. **Random Undersampling / RUS (M6):** Pemotongan acak sampel mayoritas tanpa pengembalian hingga menyamai ukuran kelas minoritas.
5. **SMOTE (M7):** Sintesis sampel minoritas berbasis interpolasi tetangga terdekat pada representasi urutan integer token.

---

## 3. Hyperparameter Selection (Milestone M2)

Pencarian grid 8 kombinasi menetapkan konfigurasi optimal yang memaksimalkan Macro F1 pada Data Validasi:
* **LSTM Units:** `128`
* **Dropout:** `0.3`
* **Learning Rate:** `0.0005`
* **Batch Size:** `16`
* **Max Sequence Length:** `128` (Post-padding)

---

## 4. Empirical Results (Hasil Empiris Distribusi Asli)

### Tabel Ringkasan Metrik Empiris (Mean ± SD Lintas 3 Random Seed)

| Strategi | Akurasi (%) | Macro F1 (%) | Presisi (%) | Recall (%) | Peringkat (Macro F1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (M3)** | **72.45 ± 1.07** | **64.95 ± 2.13** | **67.28 ± 2.09** | 63.78 ± 2.12 | **#1 (Terbaik)** |
| **ROS (M5)** | 67.05 ± 2.31 | 62.71 ± 1.16 | 63.25 ± 1.32 | 63.73 ± 0.45 | **#2** |
| **Class Weight (M4)** | 65.92 ± 3.60 | 62.70 ± 2.01 | 63.12 ± 1.19 | **65.15 ± 0.13** | **#3** |
| **RUS (M6)** | 62.95 ± 3.60 | 58.92 ± 2.33 | 59.45 ± 1.71 | 60.30 ± 2.10 | **#4** |
| **SMOTE (M7)** | 42.72 ± 1.76 | 40.76 ± 3.07 | 44.91 ± 2.90 | 44.45 ± 1.81 | **#5** |

### Perbandingan Delta Terhadap Baseline
* **ROS:** $\Delta$ Acc: -5.40 pp | $\Delta$ Macro F1: -2.24 pp | $\Delta$ Recall: -0.05 pp
* **Class Weight:** $\Delta$ Acc: -6.53 pp | $\Delta$ Macro F1: -2.25 pp | **$\Delta$ Recall: +1.37 pp**
* **RUS:** $\Delta$ Acc: -9.50 pp | $\Delta$ Macro F1: -6.03 pp | $\Delta$ Recall: -3.48 pp
* **SMOTE:** $\Delta$ Acc: -29.73 pp | $\Delta$ Macro F1: -24.19 pp | $\Delta$ Recall: -19.33 pp

---

## 5. Simulation Results (Eksperimen Simulasi Rasio Imbalance)

Dievaluasi pada 3 skenario kontrol:
* **Scenario A (1:1:1):** Data seimbang buatan (3,000 sampel).
* **Scenario B (6:3:1):** Ketimpangan moderat (5,000 sampel).
* **Scenario C (8:1:1):** Ketimpangan ekstrem (4,000 sampel).

### Tabel Komparasi Macro F1 Lintas Skenario Simulasi (%)

| Strategi | Empiris (Asli) | Skenario A (1:1:1) | Skenario B (6:3:1) | Skenario C (8:1:1) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | **64.95%** | **64.31%** | **62.88%** | 56.42% |
| **Class Weight** | 62.70% | 63.85% | 61.94% | 58.11% |
| **ROS** | 62.71% | 63.90% | 62.15% | **58.74%** |
| **RUS** | 58.92% | 63.10% | 58.42% | 52.19% |
| **SMOTE** | 40.76% | 41.52% | 39.84% | 36.20% |

---

## 6. Key Findings & Kesimpulan Ilmiah untuk Naskah Tesis

1. **Keunggulan Baseline pada Data Teks Asli:**
   Pada distribusi empiris asli, **Baseline tanpa modifikasi mencapai Macro F1 tertinggi (64.95%) dan Akurasi tertinggi (72.45%)**. Arsitektur LSTM dengan representasi embedding terlatih mampu mempelajari pola kalimat mayoritas dengan sangat baik tanpa interferensi distribusi buatan.
2. **Manfaat Nyata Class Weight untuk Sensitivitas Minoritas:**
   Meskipun Macro F1 sedikit menurun (-2.25 pp), **Class Weight secara konsisten meningkatkan Macro Recall (+1.37 pp)** dengan standar deviasi terkecil ($\pm 0.13\%$). Metode ini paling disarankan jika prioritas deteksi sentimen netral yang akurat lebih diutamakan daripada akurasi global.
3. **Peran ROS pada Ketimpangan Ekstrem (Skenario 8:1:1):**
   Pada simulasi ketimpangan ekstrem (8:1:1), Baseline mengalami degradasi tajam menjadi 56.42%. Pada kondisi ini, **Random Oversampling (ROS) dan Class Weight mengungguli Baseline** (masing-masing mencapai 58.74% dan 58.11%), membuktikan bahwa intervensi penyeimbangan data baru memberikan nilai tambah ketika rasio ketimpangan melebihi ambang batas moderat.
4. **Kerugian Fatal SMOTE pada Urutan Token Diskrit:**
   SMOTE terbukti tidak cocok untuk sequence input integer LSTM (Macro F1 anjlok ke 40.76%). Interpolasi fitur kontinu pada ID token diskret merusak struktur n-gram dan menghasilkan *pseudo-tokens* yang mengacaukan mekanisme memori LSTM.
5. **Dampak Pembuangan Data pada RUS:**
   Membuang 47.6% sampel pada RUS menurunkan keragaman kosakata secara drastis, menyebabkan penurunan performa sebesar -6.03 pp Macro F1.

---

## 7. Lokasi File Artefak & Ringkasan

* **Tabel Master:** [`Output/summary/final_results_table.csv`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/final_results_table.csv)
* **Ringkasan Empiris:** [`Output/summary/empirical_summary.csv`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/empirical_summary.csv)
* **Ringkasan Simulasi:** [`Output/summary/simulated_summary.csv`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/simulated_summary.csv)
* **Komparasi Lintas Skenario:** [`Output/summary/empirical_vs_simulated.csv`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/empirical_vs_simulated.csv)
* **Payload JSON:** [`Output/summary/final_results.json`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/final_results.json)
* **Gambar Publikasi:**
  * [`comparison_accuracy.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/comparison_accuracy.png)
  * [`comparison_macro_f1.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/comparison_macro_f1.png)
  * [`comparison_precision.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/comparison_precision.png)
  * [`comparison_recall.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/comparison_recall.png)
  * [`empirical_vs_simulated_macro_f1.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/empirical_vs_simulated_macro_f1.png)
  * [`scenario_comparison.png`](file:///d:/DATA%20SCIENCE/jokiidin/Thesis-LSTM-IndoBERT/Output/summary/scenario_comparison.png)
