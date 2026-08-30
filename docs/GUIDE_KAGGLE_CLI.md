# Panduan Training di Kaggle dari CLI (Laptop Lokal)

Tujuan: push **data** dan **notebook** dari laptop ke Kaggle, jalankan training GPU
di cloud, lalu ambil hasilnya — semuanya dari terminal, tanpa buka browser.

```
laptop (lokal)
   │  kaggle datasets version   → data masuk ke Kaggle (dataset)
   │  kaggle kernels push       → notebook masuk ke Kaggle + auto-run (GPU)
   │  kaggle kernels status     → pantau proses
   │  kaggle kernels output     → unduh hasil
   ▼
Kaggle cloud (GPU T4/P100, internet aktif, kernel private)
```

---

## 1. Prasyarat (sudah terpenuhi di laptop ini)

| Cek | Perintah | Status |
|---|---|---|
| Kaggle CLI terpasang | `kaggle --version` | 2.2.4 ✓ |
| API token | `%USERPROFILE%\.kaggle\kaggle.json` | ada ✓ |
| Akun | — | `emanuelembuaijdak` |

- Kalau `kaggle` tidak dikenali: tambahkan
  `C:\Users\emanu\AppData\Roaming\Python\Python314\Scripts` ke PATH (System Properties → Environment Variables).
- Kalau `kaggle.json` belum ada: buka https://www.kaggle.com/settings → **API** →
  **Create New Token** → letakkan file `kaggle.json` di `C:\Users\emanu\.kaggle\`.
- Kuota GPU gratis Kaggle ± 30 jam per minggu. Cek sisa kuota di halaman Settings.

---

## 2. Peta file di repo ini

| File | Fungsi |
|---|---|
| `kaggle_dataset/dataset-metadata.json` | Metadata dataset (slug: `emanuelembuaijdak/thesis-indobert-processed-data`) |
| `kaggle_dataset/split_data.pkl` | Data split: X_train_lstm/bert, X_test_lstm/bert, y_train, y_test |
| `kaggle_dataset/data_preprocessed_with_emoticon.csv` | Data bersih + emoticon |
| `kernel-metadata.json` | Metadata kernel (notebook + dataset_sources) |
| `04_model_indobertweet_lora.ipynb` | Notebook IndoBERTweet-LoRA (GPU) |
| `kaggle.yml` | Ringkasan konfigurasi proyek (referensi) |
| `.kaggle-outputs/` | Folder hasil unduhan output kernel |

> **Penting:** nama akun di metadata harus **persis** `emanuelembuaijdak`
> (bukan `emanuelembuaijidak`/`emanuelembuajidak`). Slug yang salah → error
> `403 Forbidden` / `Permission denied`.

---

## 3. Push data (dataset) dari lokal

Folder `kaggle_dataset/` sudah berisi `dataset-metadata.json` + file data.
Semua perubahan data cukup dilakukan di folder ini, lalu di-version ke Kaggle.

**Update versi dataset** (dipakai setiap data berubah):

```
kaggle datasets version -p kaggle_dataset/ -m "deskripsi perubahan"
```

**Buat dataset baru** (hanya sekali, kalau dataset belum ada di akun):

```
kaggle datasets create -p kaggle_dataset/
```

**Cek status & isi dataset:**

```
kaggle datasets status emanuelembuaijdak/thesis-indobert-processed-data
kaggle datasets files emanuelembuaijdak/thesis-indobert-processed-data
```

Syarat: slug di `dataset-metadata.json` harus sama dengan yang sudah ada di
Kaggle (untuk `version`) — saat ini sudah cocok. File `kaggle.json` / `.env`
jangan pernah ditaruh di folder dataset (akan ikut ter-upload).

---

## 4. Push notebook (kernel) dari lokal

`kernel-metadata.json` di root proyek sudah menunjuk ke
`04_model_indobertweet_lora.ipynb` dan menempelkan dataset
`thesis-indobert-processed-data` (field `dataset_sources`).

```
kaggle kernels push -p .
```

Artinya: dari root proyek, `kaggle kernels push` membaca `kernel-metadata.json`,
meng-upload notebook `code_file`, lalu **langsung menjalankannya** di GPU
(enable_gpu, enable_internet, is_private sesuai metadata).

### Memilih accelerator (GPU T4 x2, P100, TPU)

Tambahkan field `machine_shape` di `kernel-metadata.json` (CLI 2.x) — nilai valid
menurut dokumentasi resmi: `NvidiaTeslaT4` (= opsi "GPU T4 x2" di UI),
`NvidiaTeslaP100`, `Tpu1VmV38`. Bisa juga via `kaggle kernels push --accelerator <nilai>`.
Nilai tidak divalidasi klien — nilai salah baru ditolak saat runtime.

| Field | Nilai | Catatan |
|---|---|---|
| `machine_shape` | `NvidiaTeslaT4` | **Yang dipakai proyek ini** (= "GPU T4 x2" di UI Kaggle) |
| `machine_shape` | `NvidiaTeslaP100` | **Hindari** — image default Kaggle (torch cu128) tidak punya kernel Pascal (sm_60): `torch.cuda.is_available()` `True` tapi CUDA op pertama gagal `cudaErrorNoKernelImageForDevice` |
| `machine_shape` | `Tpu1VmV38` | TPU (tidak dipakai) |

Jangan mengisi nilai selain tiga di atas — server menerimanya lalu gagal saat run
(kasus versi 7: fallback ke P100 → `AcceleratorError: CUDA error: no kernel image
is available`).

Buka notebook di browser (untuk lihat log/output live):

```
https://www.kaggle.com/code/emanuelembuaijdak/thesis-indobertweet-lora-v1
```

**Push notebook lain** (misal `02_model_lstm.ipynb`): kerjakan di folder
sementara agar tidak menimpa metadata utama.

```
mkdir temp_kernel_lstm
copy 02_model_lstm.ipynb temp_kernel_lstm\
copy kernel-metadata.json temp_kernel_lstm\kernel-metadata.json
```

Edit `temp_kernel_lstm\kernel-metadata.json` (id, title, code_file), lalu:

```
kaggle kernels push -p temp_kernel_lstm
```

---

## 5. Pantau proses training

```
kaggle kernels status emanuelembuaijdak/thesis-indobertweet-lora-v1
```

Status: `running` → `complete` atau `error`. Cek list kernel sendiri:

```
kaggle kernels list --mine
```

---

## 6. Ambil hasil dari Kaggle

```
kaggle kernels output emanuelembuaijdak/thesis-indobertweet-lora-v1 -p .kaggle-outputs/
kaggle kernels pull emanuelembuaijdak/thesis-indobertweet-lora-v1 -p .kaggle-outputs/
```

- `kernels output` → hanya file hasil (CSV, PKL, plot, dsb).
- `kernels pull` → notebook lengkap dengan output sel (untuk dokumentasi tesis).

---

## 7. Penting: path data di notebook

Notebook di repo membaca file dari working directory:

```python
data_new = pd.read_pickle("data_preprocessed_with_emoticon.pkl")   # ⚠️
with open("split_data.pkl", "rb") as file: ...
```

Di Kaggle, file dari dataset berada di `/kaggle/input/thesis-indobert-processed-data/`,
**bukan** di working directory. Ada 2 hal yang perlu disesuaikan:

1. **Salin file dari `/kaggle/input`** — tambahkan sel di paling atas notebook:

   ```python
   import os, shutil
   SRC = "/kaggle/input/thesis-indobert-processed-data"
   shutil.copy(f"{SRC}/split_data.pkl", "split_data.pkl")
   ```

2. **`data_preprocessed_with_emoticon.pkl` tidak ada** (di repo maupun di
   dataset — hanya versi CSV yang tersedia). Variabel `data_new` juga tidak
   dipakai untuk pelatihan, jadi baris `pd.read_pickle(...)` sebaiknya dihapus,
   atau diganti membaca CSV:

   ```python
   data_new = pd.read_csv(f"{SRC}/data_preprocessed_with_emoticon.csv")
   ```

---

## 8. Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `403 Client Error: Forbidden` | Slug salah eja → cek `kaggle datasets list --mine` / `kaggle kernels list --mine`, samakan dengan slug di metadata |
| `Permission 'kernels.get' was denied` | Slug kernel salah / kernel private dengan nama beda → pakai slug persis dari `kaggle kernels list --mine` |
| Kernel status `ERROR` | Buka halaman kernel di browser → tab **Logs**; biasanya import hilang atau path data salah (lihat bagian 7) |
| `No GPU available` saat run | Kuota GPU mingguan habis → tunggu reset atau jadwalkan ulang |
| Dataset tidak ketemu saat run | `dataset_sources` di `kernel-metadata.json` harus persis `emanuelembuaijdak/thesis-indobert-processed-data` |
| `kaggle` not recognized | Path `...\AppData\Roaming\Python\Python314\Scripts` belum di PATH |
| Upload dataset lama | Pastikan working di folder `kaggle_dataset/` (bukan root proyek) |

---

## 9. Cheatsheet perintah

| Perintah | Fungsi |
|---|---|
| `kaggle datasets create -p kaggle_dataset/` | Upload dataset baru |
| `kaggle datasets version -p kaggle_dataset/ -m "msg"` | Update dataset |
| `kaggle datasets status <owner/slug>` | Status dataset |
| `kaggle datasets files <owner/slug>` | List file dalam dataset |
| `kaggle datasets list --mine` | List semua dataset sendiri |
| `kaggle kernels push -p <folder>` | Push + langsung jalankan notebook |
| `kaggle kernels status <owner/slug>` | Status kernel |
| `kaggle kernels output <owner/slug> -p <dir>` | Unduh file hasil |
| `kaggle kernels pull <owner/slug> -p <dir>` | Unduh notebook + output |
| `kaggle kernels list --mine` | List semua kernel sendiri |
