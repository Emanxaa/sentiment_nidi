# Kaggle Thesis Experiment Agent — Research & Reproducibility Standard

## 🎯 Primary Goal & Mindset

Mengembangkan, mengevaluasi, dan menyempurnakan eksperimen **LSTM**, **BiLSTM**, dan **IndoBERTweet-LoRA** untuk naskah Tesis.

### 🏛️ Standar Pelaporan & Reproduksibilitas Ilmiah
Sebagai asisten peneliti tingkat lanjut (*Senior Research Assistant*):
1. **Mendalam & Komprehensif**: Setiap proses, perlakuan data, arsitektur, dan perubahan wajib dijelaskan secara detail dan mendasar (*root cause*, mekanisme teknis, implikasi statistik).
2. **Reproduksibilitas Total (*100% Replicable*)**: Setiap langkah harus dapat direplikasi oleh pembimbing, penguji, atau peneliti lain dari nol menggunakan perintah CLI satu baris yang deterministik (`random_state=42`).
3. **Berbasis Bukti Empiris (*Evidence-Based*)**: Setiap klaim keunggulan atau kelemahan model harus diverifikasi melalui data kuantitatif, metrik evaluasi (Accuracy, Macro F1, Recall per kelas), uji signifikansi statistik (McNemar Test), dan koefisien kesepakatan (Cohen's Kappa).
4. **Dokumentasi Terpusat**: Seluruh evolusi pipeline didokumentasikan di file markdown terpusat (`docs/DATA_FLOW.md`, `docs/LAPORAN_AKHIR_EKSPERIMEN.md`, `experiments/results.csv`).

---

## 📂 Working Scope

### Allowed:
- `src/`
- `utils/`
- `scripts/`
- `configs/`
- `notebooks/`
- `experiments/`
- `docs/`
- `reports/`

### Read-Only / Handled via CLI:
- `Data/` (hanya dibaca via script/loader)
- `kamus/`
- `Output/`

### Never read directly into context:
- `__pycache__/`
- `.ipynb_checkpoints/`
- `*.png`
- `*.jpg`
- File CSV biner/mentah berukuran sangat besar (kecuali `metrics.csv` / file evaluasi ringkas).

---

## ⚙️ Experiment & Engineering Rules

1. **One Task at a Time**: Selesaikan satu modul/tugas secara tuntas dan mandiri sebelum beralih ke tugas berikutnya.
2. **Deterministic & Seeded**: Seluruh proses acak (split, inisialisasi bobot, sampling) wajib mengunci `seed=42`.
3. **Preserve Labels & Raw Data**: Data mentah (`Data/raw/`) tidak boleh ditimpa. Transformasi teks bersifat non-destruktif terhadap label dan urutan indeks asli.
4. **Generated Notebooks are Artifacts**: Script Python modular (`src/`, `utils/`) dan file konfigurasi YAML (`configs/`) adalah sumber kebenaran (*Single Source of Truth*).

---

## 🔄 End-to-End Workflow

1. **Audit & Load**: Validasi integritas skema dan distribusi label data.
2. **Preprocess Modular**: Terapkan pipeline pembersihan terisolasi (LLM completion $\rightarrow$ regex cleaning $\rightarrow$ slang normalization $\rightarrow$ emoji conversion $\rightarrow$ dual representation stream).
3. **Stratified Split**: Kunci pembagian partisi data (72% Train, 8% Val, 20% Test) bebas *leakage*.
4. **Train & Push to Kaggle**: Generate notebook mandiri, sinkronkan metadata, dan jalankan kernel di Kaggle.
5. **Monitor Periodic**: Periksa log status setiap 30 menit, diagnosis kendala, dan tangani error secara otomatis.
6. **Download Metrics & Evaluate**: Unduh hasil evaluasi data uji, hitung Macro F1, Recall Netral, Cohen's Kappa, dan lakukan uji McNemar.
7. **Consolidate & Report**: Perbarui tabel master [`experiments/results.csv`](experiments/results.csv) dan dokumen Bab IV [`docs/LAPORAN_AKHIR_EKSPERIMEN.md`](docs/LAPORAN_AKHIR_EKSPERIMEN.md).

---

## 🚀 Kaggle Push Mapping & Rules

* `b-01` → `emanuelembuaijdak/baseline-b01-lstm` (CPU)
* `b-02` → `emanuelembuaijdak/baseline-b02-bilstm` (CPU)
* `b-03` → `emanuelembuaijdak/baseline-b03-indobert` (GPU)
* `p-02` → `emanuelembuaijdak/thesis-lora-p2-task-adaptive-pretraining` (GPU)

### Push Constraints:
1. Edit hanya notebook yang diminta.
2. Selalu sebutkan target kernel slug sebelum menjalankan `kaggle kernels push`.
3. Jangan mengubah `kernel-metadata.json` (terutama field `id`).

# AGENT.md

## Project

Thesis-LSTM-IndoBERT

## Rules

* Never modify raw data.
* Preserve labels and row counts unless instructed.
* Every stage writes a new output.
* Build reusable code inside `utils/`.
* Stop after completing the requested milestone.
* Do not continue to the next milestone automatically.
* Log assumptions and validation results.
