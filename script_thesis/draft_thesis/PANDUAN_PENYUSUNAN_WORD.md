# PANDUAN PRAKTIS: MEMINDAHKAN HASIL DARI SCRIPT_THESIS KE MICROSOFT WORD

Dokumen ini memandu langkah demi langkah cara memindahkan seluruh tabel dan gambar hasil eksperimen dari folder `script_thesis/` ke template Microsoft Word naskah Skripsi / Tesis Anda.

---

## 1. Daftar Tabel untuk Disalin ke Word

| Nomor Tabel di Word | Judul Tabel di Word | File Sumber di `script_thesis/` |
| :--- | :--- | :--- |
| **Tabel 3.1** | Distribusi Partisi Data Latih, Validasi, dan Uji | `draft_thesis/BAB_3_METODOLOGI_PENELITIAN.md` (Tabel 3.1) |
| **Tabel 4.1** | Distribusi Frekuensi Sentimen Korpus Data V2 | `outputs/metrics/distribusi_kelas.csv` |
| **Tabel 4.2** | Hasil Evaluasi Performa Model pada Data Empiris (Tabel Master 1) | `outputs/metrics/master_model_comparison_empiris.csv` |
| **Tabel 4.3** | Perbandingan Ketahanan Model Lintas Skenario Simulasi (Tabel Master 2) | `outputs/metrics/master_model_comparison_simulasi.csv` |
| **Tabel 4.4** | Tabel Kontinjensi Uji McNemar (IndoBERTweet vs LSTM Baseline) | `draft_thesis/BAB_4_HASIL_DAN_PEMBAHASAN.md` (Subbab 4.4) |

> **Tips Cepat untuk Word**: Buka file `.csv` di atas menggunakan Microsoft Excel, lalu tekan `Ctrl+A` -> `Ctrl+C`, dan *Paste* (`Ctrl+V`) langsung ke dokumen Microsoft Word. Format tabel akan otomatis rapi sesuai style Word Anda.

---

## 2. Daftar Gambar Beresolusi Tinggi untuk Di-Insert ke Word

Semua gambar di bawah ini telah di-generate dengan resolusi publikasi standar (300 DPI) dan disimpan di folder `outputs/`:

| Nomor Gambar di Word | Keterangan Gambar | Lokasi File Gambar di `script_thesis/` |
| :--- | :--- | :--- |
| **Gambar 4.1** | Bar Plot dan Pie Chart Distribusi Sentimen Korpus | `outputs/figures/distribusi_kelas.png` |
| **Gambar 4.2** | Word Cloud Keseluruhan Korpus Tweet Banjir | `outputs/figures/wordcloud_keseluruhan.png` |
| **Gambar 4.3** | Word Cloud Per Kelas Sentimen (Negatif, Netral, Positif) | `outputs/figures/wordcloud_per_kelas.png` |
| **Gambar 4.4** | Barchart Perbandingan Multimetrik Seluruh Model | `outputs/figures/model_comparison_metrics.png` |
| **Gambar 4.5** | Grafik Garis Ketahanan Macro F1 Lintas Skenario Simulasi | `outputs/figures/simulation_resilience_comparison.png` |
| **Gambar 4.6** | Grid Confusion Matrix Komparatif Lintas Model (Empiris) | `outputs/figures/comparison_confusion_matrix_grid.png` |
| **Gambar 4.7** | Kurva Pelatihan Loss dan Akurasi LSTM (per varian 01-05) | `outputs/figures/training_curve_01_lstm_vanilla.png` dst. |
| **Gambar 4.8** | Confusion Matrix Skenario C (8:1:1) Bukti Majority Collapse | `outputs/confusion_matrix/cm_01_lstm_vanilla_scenario_811.png` |
| **Gambar 4.9** | Confusion Matrix Skenario C (8:1:1) Bukti Kebal IndoBERT | `outputs/confusion_matrix/cm_06_indobert_lora_scenario_811.png` |
| **Gambar 4.10** | Confusion Matrix IndoBERTweet-LoRA P1 Sweep Optimal | `outputs/confusion_matrix/cm_08_indobert_lora_sweep.png` |
| **Gambar 4.11** | Confusion Matrix IndoBERTweet-LoRA + TAPT Rekor Tertinggi (Empiris) | `outputs/confusion_matrix/cm_09_indobert_lora_tapt.png` |
| **Gambar 4.12** | Confusion Matrix Skenario C (8:1:1) Bukti Ketahanan Ekstrem TAPT | `outputs/confusion_matrix/cm_09_indobert_lora_tapt_scenario_811.png` |

---

## 3. Langkah Rekomendasi Saat Sidang / Bimbingan Dosen

1. **Ketika dosen bertanya mengenai proses pembersihan data**:
   Tunjukkan Notebook `00_data_preprocessing.ipynb`. Perlihatkan bahwa pembersihan regex, normalisasi slang dengan kamus, dan pemisahan data dilakukan tanpa kebocoran data (*zero data leakage*, kunci `seed=42`).
2. **Ketika dosen bertanya mengapa LSTM gagal di data tidak seimbang**:
   Tunjukkan Notebook `01_lstm_nonbalancing.ipynb` bagian Skenario C (8:1:1) dan gambar `cm_01_lstm_vanilla_scenario_811.png`. Jelaskan bahwa terjadi fenomena *Majority Collapse* di mana Recall Netral jatuh ke 0,33% karena model hanya memprediksi kelas mayoritas.
3. **Ketika dosen bertanya teknik sampling mana yang terbaik untuk LSTM**:
   Tunjukkan Notebook `03_lstm_random_oversampling.ipynb`. Jelaskan bahwa Random Oversampling (ROS) mempertahankan Macro F1 59,31% dan Recall Netral 36,42% pada rasio 8:1:1.
4. **Ketika dosen bertanya mengapa IndoBERTweet-LoRA lebih unggul**:
   Tunjukkan Notebook `06_indobert_lora.ipynb` dan Notebook `07_model_comparison.ipynb`. Jelaskan bahwa arsitektur Transformer memiliki representasi kontekstual dwiarah (*bidirectional attention*) yang memberikan kekebalan arsitektural terhadap fenomena collapse, dengan keunggulan yang terbukti signifikan secara statistik pada taraf 99,99% via Uji McNemar ($p < 0,0001$).
5. **Ketika dosen bertanya apakah penyetelan hiperparameter (tuning) sudah dicoba secara maksimal**:
   Tunjukkan Notebook `08_indobert_lora_sweep.ipynb`. Jelaskan bahwa eksperimen *Hyperparameter Sweep* sebanyak 10 uji coba telah dilakukan. Konfigurasi optimum (Trial 10) berhasil melipatgandakan Recall Netral ke 61,26% dan F1 ke 73,90%, namun akurasi mengalami batas atas (*hyperparameter ceiling*) di kisaran 78%. Hal ini membuktikan bahwa batas performa bukan berasal dari salah setting parameter, melainkan dari keterbatasan representasi kosakata umum terhadap domain banjir.
6. **Ketika dosen bertanya apa kontribusi inovasi terbaik (novelty) dari tesis ini**:
   Tunjukkan Notebook `09_indobert_lora_tapt.ipynb`. Jelaskan penerapan *Task-Adaptive Pretraining* (TAPT) melalui *Masked Language Modeling* (MLM 3 epoch) pada korpus banjir sebelum fine-tuning LoRA. Inovasi adaptasi domain tanpa supervisi ini berhasil memecahkan rekor tertinggi penelitian dengan **Akurasi 80,06%** dan **Macro F1 74,61%** (model pertama dan satu-satunya yang menembus akurasi 80%).
7. **Ketika dosen bertanya bagaimana ketahanan model TAPT pada kondisi data tidak seimbang (3 simulasi)**:
   Tunjukkan Notebook `09_indobert_lora_tapt.ipynb` Bagian 5 dan gambar `cm_09_indobert_lora_tapt_scenario_811.png`. Jelaskan bahwa TAPT membuktikan ketahanan tertinggi mutlak di seluruh simulasi: pada rasio 8:1:1, TAPT mempertahankan Macro F1 **71,95%** dan Recall Netral **39,50%**, membuktikan bahwa penguasaan leksikon banjir lokal melindungi representasi kelas minoritas dari kepunahan.


