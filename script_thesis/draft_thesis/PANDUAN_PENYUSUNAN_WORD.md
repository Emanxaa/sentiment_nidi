# PANDUAN PRAKTIS: MEMINDAHKAN HASIL DARI SCRIPT_THESIS KE MICROSOFT WORD

Dokumen ini memandu langkah demi langkah cara memindahkan seluruh tabel dan gambar hasil eksperimen dari folder `script_thesis/` ke template Microsoft Word naskah Skripsi / Tesis Anda.

---

## 1. Daftar Tabel untuk Disalin ke Word

| Nomor Tabel di Word | Judul Tabel di Word | File Sumber di `script_thesis/` |
| :--- | :--- | :--- |
| **Tabel 3.1** | Distribusi Partisi Data Latih, Validasi, dan Uji | `draft_thesis/BAB_3_METODOLOGI_PENELITIAN.md` (Tabel 3.1) |
| **Tabel 4.1** | Distribusi Frekuensi Sentimen Korpus Data V2 | `draft_thesis/BAB_4_HASIL_DAN_PEMBAHASAN.md` (Tabel 4.1) |
| **Tabel 4.2** | Hasil Evaluasi Performa Model pada Data Empiris (Tabel Master 7 Model) | `outputs/metrics/tabel_komparasi_seluruh_model.csv` |
| **Tabel 4.3** | Perbandingan Ketahanan Model Lintas Skenario Simulasi | `draft_thesis/BAB_4_HASIL_DAN_PEMBAHASAN.md` (Tabel 4.3) |
| **Tabel 4.4** | Tabel Kontinjensi & Hasil Uji McNemar (IndoBERTweet vs LSTM Baseline) | `draft_thesis/BAB_4_HASIL_DAN_PEMBAHASAN.md` (Subbab 4.4) |

> 💡 **Tips Cepat untuk Word**: Buka file `tabel_komparasi_seluruh_model.csv` menggunakan Microsoft Excel, lalu tekan `Ctrl+A` -> `Ctrl+C`, dan *Paste* (`Ctrl+V`) langsung ke dokumen Microsoft Word. Format tabel akan otomatis rapi sesuai style tabel Word Anda.

---

## 2. Daftar Gambar Beresolusi Tinggi untuk Di-Insert ke Word

Semua gambar visualisasi utama tersimpan di dalam notebook atau folder gambar:

| Nomor Gambar di Word | Keterangan Gambar | Lokasi File Gambar di `script_thesis/` |
| :--- | :--- | :--- |
| **Gambar 4.1** | Visualisasi WordCloud 4 Panel (Keseluruhan, Positif, Negatif, Netral) | Dihasilkan oleh `01_data_processing.ipynb` |
| **Gambar 4.2** | Confusion Matrix LSTM Baseline (Natural Imbalance) | Dihasilkan oleh `02_lstm_imbalance.ipynb` |
| **Gambar 4.3** | Confusion Matrix LSTM Class Weighting | Dihasilkan oleh `03_lstm_class_weight.ipynb` |
| **Gambar 4.4** | Confusion Matrix LSTM Random Oversampling (ROS) | Dihasilkan oleh `04_lstm_oversampling.ipynb` |
| **Gambar 4.5** | Confusion Matrix LSTM Random Undersampling (RUS) | Dihasilkan oleh `05_lstm_undersampling.ipynb` |
| **Gambar 4.6** | Confusion Matrix LSTM SMOTE | Dihasilkan oleh `06_lstm_smote.ipynb` |
| **Gambar 4.7** | Confusion Matrix IndoBERTweet-LoRA Vanilla | Dihasilkan oleh `07_indobert_lora.ipynb` |
| **Gambar 4.8** | Confusion Matrix TAPT IndoBERTweet-LoRA (Model Terbaik) | Dihasilkan oleh `08_tapt_indobert_lora.ipynb` |
| **Gambar 4.9** | Visualisasi 4-Panel Komparatif Seluruh Model (Train vs Test Acc, F1, Recall Netral, Generalization Gap) | Dihasilkan oleh `09_summary_model.ipynb` |

---

## 3. Panduan Jawaban Cepat Saat Sidang / Bimbingan Dosen

1. **Ketika dosen bertanya mengenai proses pembersihan data**:
   Tunjukkan Notebook `01_data_processing.ipynb`. Jelaskan bahwa rekonstruksi LLM, pembersihan regex, normalisasi slang leksikon 4.334 kata, dan konversi emoticon dilakukan secara non-destruktif dengan kunci `seed=42`.
2. **Ketika dosen bertanya mengapa akurasi Baseline LSTM (70,92%) lebih tinggi dari RUS (53,06%) dan SMOTE (64,39%)**:
   Jelaskan fenomena *The Accuracy Paradox*: Data didominasi kelas mayoritas (82% Positif/Negatif). Model baseline yang bias mayoritas mudah meraih akurasi tinggi tetapi mengalami *Majority Collapse* (Recall Netral hanya 28,81%). RUS membuang >45% data sehingga terjadi *information loss*, sedangkan SMOTE merusak tata bahasa dengan token sintetis semu. Tujuan penyeimbangan adalah menyelamatkan kelas minoritas (Recall Netral naik tajam ke 40,40% pada Class Weight dan 48,68% pada ROS).
3. **Ketika dosen bertanya model mana yang terbaik**:
   Tunjukkan Notebook `08_tapt_indobert_lora.ipynb` dan Notebook `09_summary_model.ipynb`. Jelaskan bahwa **TAPT IndoBERTweet-LoRA** adalah juara terbaik mutlak dengan **Akurasi 80,06%** dan **Macro F1 74,61%** berkat adaptasi kosakata kebencanaan via MLM 3 epoch.
4. **Ketika dosen bertanya apakah keunggulan Transformer signifikan secara statistik**:
   Tunjukkan Subbab 4.4 di `BAB_4_HASIL_DAN_PEMBAHASAN.md`. Jelaskan bahwa Uji McNemar menghasilkan $\chi^2 = 37,43$ dengan $p < 0,0001$, menolak hipotesis nol secara mutlak pada taraf signifikansi 99,99%.
