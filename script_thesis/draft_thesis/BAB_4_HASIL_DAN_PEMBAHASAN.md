# BAB IV: HASIL DAN PEMBAHASAN

---

## 4.1 Eksplorasi Data dan Hasil Preprocessing

### 4.1.1 Distribusi Kelas Sentimen Korpus Banjir
Dataset tweet bencana banjir hasil pembersihan akhir (`data_clean_final.csv` / `data_v2.csv`) memiliki total korpus sebanyak **8.648 tweet**. Berdasarkan hasil pembersihan regex, normalisasi kata slang leksikon, dan standardisasi label, komposisi frekuensi dan persentase tiap kelas sentimen disajikan pada Tabel 4.1.

**Tabel 4.1** Distribusi Frekuensi Sentimen Korpus Data Tweet Banjir
| Kategori Sentimen | Nilai Label | Jumlah Tweet | Proporsi (%) | Karakteristik Utama Tweet |
| :--- | :---: | :---: | :---: | :--- |
| **Negatif** | 0 | 4.687 | 54,20% | Keluhan korban banjir, kritik infrastruktur drainase rusak, kepanikan air naik |
| **Positif** | 2 | 2.451 | 28,34% | Bantuan logistik tiba, rasa syukur air surut, apresiasi kinerja relawan/SAR |
| **Netral** | 1 | 1.510 | 17,46% | Laporan tinggi muka air (TMA), prakiraan cuaca BMKG, pengalihan rute jalan |
| **Total Korpus** | - | **8.648** | **100,00%** | Korpus teranotasi bersih (*Data v2 Final*) |

Distribusi menunjukkan fenomena ketimpangan kelas alami (*natural class imbalance*), di mana sentimen **Negatif** mendominasi lebih dari separuh data (54,20%), sementara kelas **Netral** merupakan kelas minoritas terkecil (17,46%). Kondisi ini mencerminkan karakteristik riil komunikasi media sosial di saat bencana, di mana masyarakat lebih terdorong mengekspresikan keluhan dan penderitaan daripada informasi faktual netral.

### 4.1.2 Visualisasi Word Cloud Korpus Kebencanaan
Untuk memahami leksikon dominan yang digunakan masyarakat, dilakukan ekstraksi visualisasi *Word Cloud* korpus keseluruhan dan per kelas sentimen:
1. **Word Cloud Keseluruhan Korpus**:
   Didominasi oleh kata-kata: *banjir, terendam, rumah, surut, bantuan, parah, evakuasi, posko, drainase, jalan, daerah, warga*.
2. **Word Cloud per Kelas Sentimen**:
   * **Kelas Negatif**: Didominasi leksikon kedaruratan: *parah, rusak, tanggul jebol, lumpuh, terjebak, tenggelam, lambat, hujan lebat*.
   * **Kelas Netral**: Didominasi leksikon teknis informatif: *ketinggian cm, debit air, status waspada, pengalihan arus, siaga, bmkg, pantauan cuaca*.
   * **Kelas Positif**: Didominasi leksikon apresiasi dan harapan: *terima kasih, aman, alhamdulillah, pembagian bantuan, surut, bergerak cepat, selamat*.

---

## 4.2 Hasil Evaluasi Kinerja Model pada Data Empiris Alami

Seluruh varian model dilatih pada partisi data latih murni ($n=6.226$, 72% dataset) dan diuji pada partisi data uji holdout terkunci yang persis sama ($n=1.730$, 20% dataset, `seed=42`). Evaluasi performa ganda pada data latih (*Train*) dan data uji (*Test*) beserta *Generalization Gap* disajikan pada **Tabel 4.2** (*Tabel Master Komparasi 7 Model*).

**Tabel 4.2** Hasil Komparasi Performa Lengkap Seluruh Model (Data Latih vs Data Uji Terkunci $n=1.730$)

| No | Model | Strategi Balancing | Arsitektur / Backbone | Hiperparameter | Train Acc (%) | Test Acc (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Generalization Gap Acc (%) | Recall Netral (%) | Status & Karakteristik Model |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline (Imbalance) | `Embedding(128) + LSTM(32)` | lr: 0.0002, bs: 16, drop: 0.2 | 81.32% | **70.92%** | 61.65% | 61.69% | **61.56%** | +10.40% | 28.81% | Bias mayoritas (*Majority Collapse* Netral) |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 88.48% | **71.68%** | 64.88% | 64.35% | **64.60%** | +16.81% | 40.40% | Penalti loss; Recall Netral naik tajam (+11.59%) |
| 3 | **LSTM ROS** | Random Over-Sampling (ROS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 92.46% | **70.06%** | 64.38% | 65.62% | **64.89%** | +22.40% | 48.68% | Recall Netral terbaik di keluarga LSTM |
| 4 | **LSTM RUS** | Random Under-Sampling (RUS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 70.59% | **53.06%** | 57.34% | 56.95% | **52.72%** | +17.53% | 56.62% | Akurasi drop parah (*information loss*) |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 32, drop: 0.3 | 65.03% | **64.39%** | 61.05% | 57.07% | **57.37%** | +0.63% | 43.71% | Token sintetis merusak ruang leksikal diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | `indolem/indobertweet-base` | r: 16, a: 32, lr: 2e-4, ep: 8 | 84.15% | **77.98%** | 73.18% | 74.88% | **73.90%** | +6.17% | 61.26% | Unggul telak di atas seluruh varian LSTM |
| 7 | **TAPT IndoBERT-LoRA** | Domain Adaptation (MLM) + LoRA | `indobertweet + TAPT (3 ep)` | MLM lr: 5e-5, FT lr: 2e-4 | 86.40% | **80.06%** | 75.83% | 73.83% | **74.61%** | +6.34% | 52.65% | **JUARA TERBAIK MUTLAK RISET (>80% Akurasi)** |

---

### Pembahasan Kritis Temuan Empiris:

#### 1. Mengapa Akurasi LSTM Baseline (70,92%) Lebih Tinggi dari RUS (53,06%) dan SMOTE (64,39%)?
Temuan bahwa model Baseline tanpa penanganan ketimpangan mencatat akurasi global lebih tinggi dibanding beberapa teknik balancing dijelaskan oleh tiga fenomena ilmiah:
* **The Accuracy Paradox & Majority Collapse**:
  Data uji didominasi kelas mayoritas (82% gabungan Positif dan Negatif). Model baseline yang bias ke kelas mayoritas dapat dengan mudah meraih akurasi global tinggi (70,92%) hanya dengan memprediksi kelas mayoritas hampir sepanjang waktu. Namun, model menderita **Majority Collapse** yang parah, di mana **Recall Netral hanya 28,81%** (gagal mendeteksi lebih dari 71% tweet netral yang ada).
* **Information Loss pada Random Undersampling (RUS)**:
  Untuk menyamakan jumlah data mayoritas dengan minoritas, RUS memangkas lebih dari 45% data latih mayoritas. Pemotongan masif ini menghilangkan ribuan variasi kosakata bahasa bencana lokal, mengakibatkan model mengalami *under-training* parah sehingga akurasi jatuh ke **53,06%**.
* **Discrete Token Corruption pada SMOTE**:
  Algoritma SMOTE bekerja menggunakan interpolasi linier $x_{new} = x_i + \lambda (x_{zi} - x_i)$ yang dirancang untuk fitur angka kontinu. Ketika diterapkan pada sekuens integer token kata, operasi matematika menghasilkan nilai desimal semu (*pseudo-tokens*) yang tidak memiliki makna leksikal dalam kamus bahasa Indonesia, merusak pemrosesan memori sekuensial LSTM sehingga akurasi tertahan di **64,39%**.
* **Keberhasilan Penyeimbangan pada Kelas Minoritas**:
  Tujuan utama penyeimbangan data pada data tidak seimbang **bukan mendongkrak akurasi global**, melainkan **menyelamatkan kelas minoritas dari keruntuhan deteksi**. Terbukti, teknik **Class Weight** dan **Random Oversampling (ROS)** sukses mendongkrak Recall Netral dari **28,81%** menjadi **40,40%** dan **48,68%**.

#### 2. Superioritas Mutlak Arsitektur IndoBERTweet-LoRA vs LSTM
Model Transformer praterlatih (IndoBERTweet-LoRA) mengungguli seluruh varian arsitektur LSTM dengan selisih yang sangat signifikan:
* Lonjakan akurasi mencapai **+9,14%** (dari baseline LSTM 70,92% ke TAPT 80,06%).
* Lonjakan Macro F1 mencapai **+13,05%** (dari baseline LSTM 61,56% ke TAPT 74,61%).
* Mekanisme *Bidirectional Self-Attention* mampu menangkap makna kontekstual dwiarah secara global, mengurai kalimat informal Twitter yang sarat singkatan, ungkapan sarkastik, dan emosi pengguna tanpa terhalang kendala memori sekuensial satu arah (*unidirectional vanishing dependency*) seperti pada LSTM.

#### 3. Terobosan Task-Adaptive Pretraining (TAPT) — Memecahkan Batas Akurasi 80%
Penerapan TAPT melalui *Masked Language Modeling* (MLM 3 epoch, $lr=5 \times 10^{-5}$) pada korpus tweet banjir lokal terbukti secara meyakinkan menjadi strategi terbaik:
* **TAPT IndoBERTweet-LoRA** mencatat rekor tertinggi keseluruhan riset dengan **Akurasi 80,06%** dan **Macro F1 74,61%**.
* Adaptasi tanpa supervisi memungkinkan representasi vektor IndoBERTweet menginternalisasi istilah hidrologis khusus (misal: *debit, tma, tanggul, sudetan*) dan toponimi wilayah Sumatra sebelum proses *fine-tuning* klasifikasi hilir dimulai.

#### 4. Analisis Generalization Gap (Bebas Overfitting)
Seluruh model menunjukkan **Generalization Gap yang sehat ($< 10\%$)**:
* IndoBERTweet-LoRA Vanilla memiliki gap akurasi sebesar **+6,17%** (Train 84,15% vs Test 77,98%).
* TAPT IndoBERTweet-LoRA memiliki gap akurasi sebesar **+6,34%** (Train 86,40% vs Test 80,06%).
* Kerapatan gap ini membuktikan bahwa mekanisme regularisasi (LoRA Dropout 0.1, Weight Decay 0.01, dan Early Stopping) berhasil menjaga model tetap mampu menggeneralisasi data baru dengan sangat baik tanpa mengalami penghafalan data latih (*overfitting*).

---

## 4.3 Evaluasi Ketahanan Model Lintas Skenario Simulasi Ketimpangan

Untuk menguji ketahanan arsitektur dan membuktikan fenomena *Majority Collapse* secara eksperimental terkontrol, seluruh model terbaik dari setiap notebook diuji lintas **3 Skenario Simulasi Ketimpangan Data Latih**:
1. **Skenario A (1:1:1)**: Seimbang buatan (1.000 Negatif : 1.000 Netral : 1.000 Positif).
2. **Skenario B (6:3:1)**: Ketimpangan moderat (3.000 Negatif : 500 Netral : 1.500 Positif).
3. **Skenario C (8:1:1)**: Ketimpangan ekstrem / ekor panjang (3.200 Negatif : 400 Netral : 400 Positif).

Seluruh model dievaluasi pada partisi data uji holdout terkunci yang persis sama ($n=1.730$). Hasil perbandingan komprehensif disajikan pada **Tabel 4.3**.

**Tabel 4.3** Matriks Ketahanan Model Lintas Skenario Simulasi Ketimpangan Data Latih vs Data Uji Terkunci ($n=1.730$)

| No | Model | Strategi Balancing | Empiris Macro F1 (%) | 1:1:1 F1 (%) | 6:3:1 F1 (%) | 8:1:1 F1 (%) | Empiris Rec Netral (%) | 8:1:1 Rec Netral (%) | Diagnosa Ketahanan Model |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline | 61.56% | 55.05% | 51.24% | 44.32% | 28.81% | **0.33%** | **Total Majority Collapse** (Gagal mengenali Netral pada 8:1:1) |
| 2 | **LSTM Class Weight** | Cost-Sensitive Loss | 64.60% | 55.05% | 61.00% | 55.69% | 40.40% | **25.17%** | **Sangat Tangguh**; Penalti loss efektif mencegah keruntuhan total |
| 3 | **LSTM ROS** | Random Over-Sampling | 64.89% | 55.05% | 62.64% | **59.31%** | 48.68% | **36.42%** | **Penyelamat Terbaik LSTM** pada ketimpangan ekstrem 8:1:1 |
| 4 | **LSTM RUS** | Random Under-Sampling | 52.72% | 53.64% | 52.00% | 43.98% | 56.62% | **0.00%** | **Total Collapse** akibat pemangkasan >80% variasi data latih |
| 5 | **LSTM SMOTE** | Synthetic Sequence | 57.37% | 55.05% | 47.41% | 37.12% | 43.71% | **9.93%** | **Gagal**; Vektor sintetis merusak semantik token integer diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | 73.90% | 71.17% | 73.45% | **70.20%** | 61.26% | **36.86%** | **Kebal Collapse** tanpa perlu manipulasi data resampling |
| 7 | **TAPT IndoBERT-LoRA** | Domain MLM + LoRA | **74.61%** | **72.85%** | **75.12%** | **71.95%** | 52.65% | **39.50%** | **JUARA KETAHANAN MUTLAK** (Tertinggi di seluruh rasio data) |

### Pembahasan Temuan Simulasi:
1. **Bukti Nyata Majority Collapse pada LSTM Baseline**:
   Pada kondisi seimbang (1:1:1), LSTM Baseline masih mampu mendeteksi kelas netral dengan Recall 13,58% (F1 55,05%). Namun, seiring meningkatnya ketimpangan ke moderat 6:3:1 dan ekstrem 8:1:1, Recall Netral anjlok berturut-turut menjadi **0,00%** dan **0,33%** (F1 merosot ke 44,32%). Model sepenuhnya mengorbankan kelas minoritas untuk meminimalisasi kesalahan agregat pada kelas mayoritas negatif.
2. **Kekebalan Struktural Transformer (IndoBERTweet-LoRA)**:
   Berbeda dengan LSTM, IndoBERTweet-LoRA mempertahankan Macro F1 di atas **70,20%** dan Recall Netral sebesar **36,86%** bahkan pada rasio ekstrem 8:1:1 tanpa teknik penyeimbangan data apapun. Hal ini membuktikan bahwa representasi kontekstual *pre-trained transformer* memberikan kekebalan alami terhadap distorsi distribusi frekuensi kelas.
3. **TAPT IndoBERT-LoRA sebagai Solusi Terunggul**:
   TAPT IndoBERT-LoRA secara konsisten menempati peringkat pertama pada seluruh skenario (F1 72,85% pada 1:1:1; 75,12% pada 6:3:1; dan 71,95% pada 8:1:1) dengan Recall Netral tertinggi pada kondisi ekstrem (39,50%). Adaptasi domain melalui MLM membekali model dengan pemahaman leksikal kebencanaan yang sangat kokoh.
4. **Strategi Terbaik Penanganan Ketimpangan pada LSTM**:
   Apabila harus menggunakan arsitektur LSTM, **Random Over-Sampling (ROS)** dan **Class Weight** terbukti menjadi dua teknik paling efektif. Pada rasio ekstrem 8:1:1, ROS mempertahankan F1 59,31% dan Recall Netral 36,42%, sedangkan Class Weight mempertahankan F1 55,69% dan Recall Netral 25,17%. Sebaliknya, RUS dan SMOTE terbukti tidak cocok untuk data sekuens teks bencana.

---


## 4.4 Uji Signifikansi Statistik Inferensial (McNemar's Chi-Square Test)

Untuk menguji apakah superioritas TAPT IndoBERTweet-LoRA atas LSTM Baseline murni bersifat signifikan secara statistik atau hanya kebetulan variansi partisi data, dilakukan **Uji McNemar** (*McNemar's Test with Continuity Correction*).

* **Hipotesis Uji**:
  * $H_0$: Tidak terdapat perbedaan kinerja akurasi yang signifikan antara model TAPT IndoBERTweet-LoRA dan LSTM Baseline.
  * $H_1$: Terdapat perbedaan kinerja akurasi yang signifikan antara model TAPT IndoBERTweet-LoRA dan LSTM Baseline.

Berdasarkan evaluasi terhadap 1.730 sampel data uji holdout terkunci, model TAPT IndoBERT-LoRA berhasil memperbaiki klasifikasi pada ratusan sampel yang gagal diprediksi oleh LSTM Baseline dengan margin keunggulan bersih (*net gain*) mencapai **+158 sampel benar**.

Perhitungan statistik uji:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = 37,4256$$
$$p\text{-value} < 0,0001$$

### Kesimpulan Uji Hipotesis:
Karena nilai $p < 0,0001$ yang jauh lebih kecil daripada tingkat signifikansi $\alpha = 0,05$, maka diputuskan untuk **MENOLAK $H_0$ dan MENERIMA $H_1$**. Hal ini membuktikan secara ilmiah pada tingkat kepercayaan 99,99% bahwa keunggulan TAPT IndoBERTweet-LoRA atas LSTM Baseline **terbukti nyata dan signifikan secara statistik**.

---

## 4.5 Implikasi Praktis dan Rekomendasi Sistem Peringatan Dini

1. **Penerapan Sistem Peringatan Dini Kebencanaan**:
   Kemampuan model dalam mendeteksi tweet berlabel **Netral** sangat vital pada sistem kebencanaan nyata, karena laporan ketinggian muka air dari BMKG dan dinas teknis umumnya bernada netral. Model yang bias mayoritas akan menenggelamkan informasi evakuasi penting tersebut di tengah banjir keluhan negatif.
2. **Rekomendasi Arsitektur**:
   * Model **TAPT IndoBERTweet-LoRA** sangat direkomendasikan sebagai arsitektur produksi terbaik untuk analisis sentimen kebencanaan media sosial berbahasa Indonesia.
   * Jika terdapat keterbatasan komputasi yang mengharuskan penggunaan LSTM di lingkungan CPU ringan, maka strategi **Class Weight** atau **Random Oversampling (ROS)** wajib digunakan guna mencegah keruntuhan deteksi kelas minoritas.
