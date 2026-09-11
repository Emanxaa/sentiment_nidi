# 📁 Panduan Struktur & Versi Dataset (Data v1 vs Data v2)

Direktori ini memuat seluruh tingkatan data yang digunakan dalam penelitian tesis **Analisis Sentimen Bencana Banjir Sumatra**:

---

## 1. Perbandingan Cepat: Data v1 vs Data v2

| Aspek | Data v1 (Dataset Lama) | Data v2 (Dataset Bersih Final / Resmi) |
| :--- | :--- | :--- |
| **Lokasi File** | [`Data/data_preprocessed_with_emoticon.csv`](data_preprocessed_with_emoticon.csv) | [`Data/processed/data_clean_final.csv`](processed/data_clean_final.csv) |
| **Jumlah Baris** | 8.650 baris | **8.648 baris** (setelah audit & deduplikasi bersih) |
| **Penanganan Truncation** | Teks terpotong akibat UI Twitter dibiarkan apa adanya | Direkonstruksi secara kontekstual berbasis LLM (`Data/interim/llm_completed.csv`) |
| **Pembersihan Regex** | Regex dasar | Regex sanitasi bertahap (URL, mention, hashtag, karakter non-ASCII) |
| **Normalisasi Slang/Alay** | Parsial | Normalisasi penuh dengan kamus resmi 4.334 leksikon (`kamus/colloquial-indonesian-lexicon.csv`) |
| **Penanganan Emoticon** | Dibiarkan berupa karakter simbol | Diterjemahkan menjadi kata emosi sentimen bahasa Indonesia |
| **Penggunaan** | Rujukan eksperimen historis (*legacy*) | **Digunakan oleh SELURUH model utama (`02` s.d. `08`)** |

---

## 2. Struktur Lengkap Folder `Data/`

```text
Data/
├── raw/
│   └── banjir.csv                      # Data mentah scraping Twitter asli (8.650 baris)
├── data_banjir.csv                     # Salinan identik data mentah
├── data_preprocessed_with_emoticon.csv # [DATA V1] Hasil preprocessing awal
├── interim/                            # Data transisi pipeline Data-Centric AI
│   ├── audit.csv                       # Log audit anomali & deteksi 402 tweet terpotong
│   ├── llm_completed.csv               # Hasil rekonstruksi teks terpotong via LLM
│   └── regex_clean.csv                 # Teks setelah pembersihan regex
├── processed/                          # [DATA V2] DATASET RESMI UTAMA
│   ├── data_clean_final.csv            # >>> File utama yang dibaca seluruh notebook (8.648 baris) <<<
│   ├── banjir_processed_v2.csv         # File dataset terproses v2 alternatif
│   └── split_data_v2.pkl               # Objek partisi Stratified Train, Val, Test
└── simulated/                          # Dataset pengujian rasio ketimpangan kelas buatan
    ├── scenario_111.csv                # Skenario 1:1:1 (Seimbang sempurna)
    ├── scenario_631.csv                # Skenario 6:3:1 (Ketimpangan moderat)
    └── scenario_811.csv                # Skenario 8:1:1 (Ketimpangan ekstrem)
```

---

## 3. Distribusi Kelas Data v2 (`data_clean_final.csv`)

* **Total Sampel**: 8.648 tweet
* **Negatif (0)**: 4.687 tweet (54,20%)
* **Positif (2)**: 2.451 tweet (28,34%)
* **Netral (1)**: 1.510 tweet (17,46%) — *Kelas minoritas*

### Pembagian Partisi (*Zero Leakage*, Seed 42):
* **Data Latih (Train Set - 72%)**: 6.226 tweet
* **Data Validasi (Val Set - 8%)**: 692 tweet (10% dari data latih)
* **Data Uji Terkunci (Test Set - 20%)**: 1.730 tweet (digunakan menguji seluruh model)
