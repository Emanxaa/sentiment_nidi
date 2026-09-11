# 📖 Panduan Folder `draft_thesis/` — Draf Naskah Tesis Siap Salin

Selamat datang di direktori **`draft_thesis/`**. Folder ini disediakan secara khusus sebagai **jembatan langsung antara hasil komputasi/eksperimen kode dengan naskah akademik Tesis/Skripsi Anda**.

---

## 🎯 1. Fungsi Utama Folder `draft_thesis/`

Banyak peneliti atau mahasiswa menghadapi kendala saat harus menerjemahkan angka-angka dari Jupyter Notebook ke dalam naskah ilmiah skripsi. Folder ini mengatasi kendala tersebut dengan menyediakan **draf bab naskah tesis lengkap yang sudah diformat secara akademis**:
1. **Bebas Kerumitan Format**: Seluruh tabel metrik (Tabel 4.1 Empiris, Tabel 4.2 Matriks Evaluasi Lengkap, Tabel 4.3 Evaluasi Simulasi Train vs Test) telah disajikan dalam format Markdown standar yang rapi.
2. **Pembahasan Saintifik Mendalam**: Bukan sekadar menampilkan angka, melainkan memuat penjelasan kausalitas (*root cause*), dinamika fungsi rugi (*loss function*), representasi semantik (*contextual embedding*), dan implikasi praktis mitigasi bencana banjir.
3. **Siap Salin ke Microsoft Word**: Format heading, bullet points, persamaan matematika LaTeX, dan kutipan tabel dirancang agar dapat langsung disalin ke templat naskah tesis kampus.

---

## 📂 2. Struktur & Deskripsi Setiap Berkas

| Nama Berkas | Fungsi Utama & Isi Dokumen | Rujukan Sumber Kode |
| :--- | :--- | :--- |
| **`BAB_3_METODOLOGI_PENELITIAN.md`** | **Draf Lengkap Bab III Tesis**: Memuat diagram alir data-centric, formulasi matematis (*LSTM cell equations, Class Weight loss penalty, SMOTE linear interpolation, LoRA low-rank decomposition, TAPT Masked Language Modeling*), skema partisi stratified 72:8:20 (Seed=42), serta protokol evaluasi ganda Train vs Test. | [`01_data_processing.ipynb`](../01_data_processing.ipynb) |
| **`BAB_4_HASIL_DAN_PEMBAHASAN.md`** | **Draf Lengkap Bab IV Tesis**: Memuat pembahasan hasil eksperimen komprehensif: <br>• **Tabel 4.1**: Ringkasan Hasil Empiris ($n=8.650$, Test Set $n=1.730$).<br>• **Tabel 4.2**: Matriks Evaluasi Lengkap (Precision, Recall, F1 per kelas).<br>• **Tabel 4.3**: Evaluasi Ganda Ketahanan 3 Skenario Simulasi (1:1:1, 6:3:1, 8:1:1).<br>• **Pembahasan Saintifik**: *The Accuracy Paradox*, *Majority Collapse*, kegagalan RUS akibat *Vocabulary Loss*, rusaknya token diskrit pada SMOTE, stabilitas generalisasi LoRA, dan adaptasi domain TAPT. | [`02`](../02_lstm_imbalance.ipynb) s.d. [`08`](../08_tapt_indobert_lora.ipynb), [`09_summary_model.ipynb`](../09_summary_model.ipynb) |
| **`PANDUAN_PENYUSUNAN_WORD.md`** | **Panduan Praktis Konversi ke MS Word**: Langkah-demi-langkah menyalin berkas Markdown ke Microsoft Word / Google Docs tanpa merusak tata letak tabel atau rumus matematika. | Panduan Operasional Penulisan |
| **`BAB_4_HASIL_DAN_PEMBAHASAN.html`** | Berkas tampilan web HTML hasil render Bab IV yang dapat dibuka langsung di Google Chrome atau browser lainnya untuk membaca naskah secara nyaman. | Tampilan Pratinjau Cepat |

---

## 🔗 3. Hubungan Tabel Naskah Tesis dengan Notebook

Agar Anda dan penguji dapat menelusuri keaslian data (*audit trail* 100% transparan), berikut adalah pemetaan tabel tesis ke notebook asalnya:

| Tabel di Naskah Tesis | Nama Tabel di Bab IV | Notebook Rujukan Utama | Berkas CSV Sumber |
| :---: | :--- | :--- | :--- |
| **Tabel 4.1** | Hasil Evaluasi Model pada Data Alami (Empiris) | [`09_summary_model.ipynb`](../09_summary_model.ipynb) | `outputs/metrics/master_model_comparison_empiris.csv` |
| **Tabel 4.2** | Matriks Evaluasi Lengkap Presisi, Recall, F1 | [`02`](../02_lstm_imbalance.ipynb) s.d. [`08`](../08_tapt_indobert_lora.ipynb) | `outputs/metrics/tabel_komparasi_seluruh_model.csv` |
| **Tabel 4.3** | Ketahanan Model pada 3 Skenario Simulasi (Train vs Test) | [`09_summary_model.ipynb`](../09_summary_model.ipynb) | `outputs/metrics/tabel_crosscheck_simulasi.csv` |

---

## ✍️ 4. Cara Menggunakan Draf Ini untuk Naskah Skripsi Anda

1. **Buka file `BAB_4_HASIL_DAN_PEMBAHASAN.md`**:
   - Salin teks bab ke templat skripsi Word kampus Anda.
   - Tabel-tabel Markdown dapat langsung di-*copy-paste* ke Word dan akan otomatis terkonversi menjadi tabel Word.
2. **Gunakan Gambar dari `outputs/figures/`**:
   - Gambar Confusion Matrix, WordCloud, dan Grafik Komparasi tersedia dalam format PNG beresolusi tinggi di folder [`outputs/figures/`](../outputs/figures/).
3. **Pahami Argumen Utama Saat Ujian / Bimbingan**:
   - Jika dosen bertanya *"Mengapa akurasi LSTM Baseline lebih tinggi dari SMOTE?"*, buka Subbab 4.2.1 (*The Accuracy Paradox*).
   - Jika dosen bertanya *"Mengapa SMOTE gagal pada teks?"*, buka Subbab 4.2.3 (*Discrete Token Space Corruption*).
   - Jika dosen bertanya *"Mengapa IndoBERTweet-LoRA kebal majority collapse?"*, buka Subbab 4.3.2 (*Bidirectional Self-Attention & Domain MLM*).
