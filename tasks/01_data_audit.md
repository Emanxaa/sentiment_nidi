# Task 01 — Data Audit & Candidate Detection

## Goal

Audit `banjir.csv` dan tambahkan indikator (flags) untuk mengidentifikasi tweet yang berpotensi membutuhkan LLM Completion atau preprocessing lanjutan.

Jangan melakukan cleaning pada tahap ini.

---

## Input

`Data/raw/banjir.csv`

Gunakan kolom berikut:

* `text`
* `clean_text`
* `processed_text`
* `sentimen`
* `label`

---

## Tugas

### 1. Audit Dataset

Laporkan:

* jumlah baris
* jumlah kolom
* missing value per kolom
* distribusi label
* jumlah duplikasi berdasarkan `text`

Simpan ke:

`Data/interim/audit_report.md`

---

### 2. Tambahkan Kolom Audit

Buat kolom boolean berikut.

| Kolom            | Kondisi                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| `has_truncation` | mengandung `Tampilkan lebih banyak`, `View a thread`, `Lihat selengkapnya`, atau diakhiri `...` |
| `has_mention`    | mengandung `@username`                                                                          |
| `has_hashtag`    | mengandung `#kata`                                                                              |
| `has_url`        | mengandung URL                                                                                  |
| `has_unicode`    | mengandung karakter Unicode tidak lazim seperti `Â`, `𝗗`, dll.                                 |
| `has_engagement` | mengandung pola `2 rb`, `35 rb`, `1 jt`, angka engagement di akhir tweet                        |
| `has_html`       | mengandung `&amp;`, `<br>`                                                                      |

Semua kolom bertipe boolean (`True/False`).

---

### 3. Buat Ringkasan

Hitung jumlah baris untuk setiap kategori.

Contoh:

| Flag           | Jumlah |
| -------------- | ------ |
| has_truncation | xxx    |
| has_hashtag    | xxx    |
| has_mention    | xxx    |
| has_url        | xxx    |

---

### 4. Simpan Dataset Audit

Buat file:

`Data/interim/audit.csv`

Dataset ini harus berisi seluruh kolom asli ditambah kolom audit.

---

## Acceptance Criteria

* Jumlah baris sama dengan dataset asli.
* Kolom `label` dan `sentimen` tidak berubah.
* Tidak ada teks yang dimodifikasi.
* Semua flag berhasil dihitung.
* `audit_report.md` berhasil dibuat.
