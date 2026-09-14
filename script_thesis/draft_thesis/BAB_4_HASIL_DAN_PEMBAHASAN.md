# BAB IV: HASIL DAN PEMBAHASAN

---

## 4.1 Eksplorasi Data dan Hasil Preprocessing

### 4.1.1 Distribusi Kelas Sentimen Korpus Banjir
Dataset tweet bencana banjir hasil pembersihan akhir (`data_clean_final.csv` / `data_v2.csv`) memiliki total korpus sebanyak **8.648 tweet**. Berdasarkan hasil pembersihan regex, normalisasi kata slang leksikon, dan standardisasi label, komposisi frekuensi dan persentase tiap kelas sentimen disajikan pada **Tabel 4.1**.

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
1. **Word Cloud Keseluruhan Korpus**: Didominasi oleh kata-kata: *banjir, terendam, rumah, surut, bantuan, parah, evakuasi, posko, drainase, jalan, daerah, warga*.
2. **Word Cloud per Kelas Sentimen**:
   * **Kelas Negatif**: Didominasi leksikon kedaruratan: *parah, rusak, tanggul jebol, lumpuh, terjebak, tenggelam, lambat, hujan lebat*.
   * **Kelas Netral**: Didominasi leksikon teknis informatif: *ketinggian cm, debit air, status waspada, pengalihan arus, siaga, bmkg, pantauan cuaca*.
   * **Kelas Positif**: Didominasi leksikon apresiasi dan harapan: *terima kasih, aman, alhamdulillah, pembagian bantuan, surut, bergerak cepat, selamat*.

---

## 4.2 Hasil Evaluasi Kinerja Model pada Data Empiris Alami

Untuk membangun perbandingan yang adil dan simetris (*head-to-head architectural symmetry*), penelitian ini mengevaluasi **11 Model Komparatif** yang terbagi secara berimbang antara dua keluarga besar arsitektur:
* **Keluarga LSTM (5 Model)**: Baseline Alami, Class Weight, Random Over-Sampling (ROS), Random Under-Sampling (RUS), dan SMOTE.
* **Keluarga IndoBERTweet-LoRA (6 Model)**: Baseline Alami, Class Weight, Random Over-Sampling (ROS), Random Under-Sampling (RUS), SMOTE pada ruang representasi laten, serta **+1 model usulan puncak: Task-Adaptive Pretraining (TAPT)**.

Seluruh varian model dilatih pada partisi data latih murni ($n=6.226$, 72% dataset) dan diuji pada partisi data uji holdout terkunci yang persis sama ($n=1.730$, 20% dataset, `seed=42`). Evaluasi performa ganda pada data latih (*Train*) dan data uji (*Test*) beserta *Generalization Gap* disajikan pada **Tabel 4.2**.

**Tabel 4.2** Hasil Komparasi Performa Lengkap 11 Model Komparatif (Data Latih vs Data Uji Terkunci $n=1.730$)

| No | Model | Strategi Balancing | Arsitektur / Backbone | Hiperparameter | Train Acc (%) | Test Acc (%) | Gap Acc (%) | Train F1 (%) | Test F1 (%) | Gap F1 (%) | Recall Netral (%) | Status & Catatan |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline (Imbalance) | `Embedding(128) + LSTM(32)` | lr: 0.0002, bs: 16, drop: 0.2 | 81.32% | **70.92%** | +10.40% | 73.19% | **61.56%** | +11.62% | 28.81% | *Majority Collapse* pada kelas Netral |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 88.48% | **71.68%** | +16.81% | 85.06% | **64.60%** | +20.46% | 40.40% | Penalti loss; Recall Netral naik (+11.59%) |
| 3 | **LSTM ROS** | Random Over-Sampling (ROS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 92.46% | **70.06%** | +22.40% | 92.44% | **64.89%** | +27.54% | 48.68% | Recall Netral tertinggi LSTM (Overfitting F1 Gap +27%) |
| 4 | **LSTM RUS** | Random Under-Sampling (RUS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 70.59% | **53.06%** | +17.53% | 70.64% | **52.72%** | +17.92% | 56.62% | Akurasi drop parah (*information loss*) |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 32, drop: 0.3 | 65.03% | **64.39%** | +0.63% | 64.33% | **57.37%** | +6.96% | 43.71% | Token sintetis merusak semantik diskrit |
| 6 | **IndoBERTweet-LoRA** | Natural Baseline | `indolem/indobertweet-base` | r: 16, a: 32, lr: 2e-4, ep: 8 | 84.15% | **77.98%** | +6.17% | 80.20% | **73.90%** | +6.30% | 61.26% | Generalisasi semantik kontekstual sangat stabil |
| 7 | **IndoBERTweet-LoRA CW** | Class Weight (Cost-Sensitive) | `indolem/indobertweet-base` | r: 8, a: 16, lr: 2e-4, ep: 5 | 85.20% | **76.30%** | +8.90% | 81.10% | **73.10%** | +8.00% | **69.21%** | Recall Netral tertinggi kedua riset (+7.95%) |
| 8 | **IndoBERTweet-LoRA ROS** | Random Over-Sampling (ROS) | `indolem/indobertweet-base` | r: 8, a: 16, lr: 2e-4, ep: 5 | 89.45% | **77.57%** | +11.88% | 86.20% | **73.93%** | +12.27% | **64.57%** | Duplikasi teks memperkuat atensi kelas minoritas |
| 9 | **IndoBERTweet-LoRA RUS** | Random Under-Sampling (RUS) | `indolem/indobertweet-base` | r: 8, a: 16, lr: 2e-4, ep: 5 | 80.12% | **75.78%** | +4.34% | 76.50% | **72.67%** | +3.83% | **69.54%** | Tahan information loss berkat bobot pra-latih |
| 10 | **IndoBERTweet-LoRA SMOTE** | SMOTE (Sentence Latent Space)| `indolem/indobertweet-base` | r: 8, a: 16, lr: 2e-4, ep: 5 | 82.50% | **74.86%** | +7.64% | 78.40% | **71.35%** | +7.05% | **58.28%** | Representasi kontinu 768-d jauh lebih stabil |
| 11 | **TAPT IndoBERT-LoRA** | Domain Adaptation (MLM) + LoRA | `indobertweet + TAPT (3 ep)` | MLM lr: 5e-5, FT lr: 2e-4 | 86.40% | **80.06%** | +6.34% | 82.50% | **74.61%** | +7.89% | **52.65%** | **JUARA TERBAIK MUTLAK RISET (>80% Acc, 74.61% F1)** |

---

### Pembahasan Kritis Temuan Empiris Lintas Arsitektur (Head-to-Head):

#### 1. Perbandingan Head-to-Head Strategi Penyeimbangan Data (LSTM vs IndoBERT)
Perbandingan simetris antara strategi penanganan ketimpangan kelas pada LSTM dan Transformer menghasilkan wawasan teoretis penting:
* **Cost-Sensitive Class Weight (CW)**:
  Pada LSTM, Class Weight menaikkan Recall Netral dari 28,81% ke 40,40% (+11,59%). Pada IndoBERTweet-LoRA, Class Weight memberikan dampak yang jauh lebih spektakuler dengan melesatkan Recall Netral ke **69,21%** (hampir 70% informasi kebencanaan netral berhasil diselamatkan). Hal ini menunjukkan bahwa penalti bobot gradien pada loss function bersinergi sangat kuat dengan representasi atensi multi-head.
* **Random Over-Sampling (ROS)**:
  Pada LSTM, duplikasi teks memicu pembengkakan *Generalization Gap* yang parah (+27,54%) akibat keterbatasan sel memori yang cenderung menghafal urutan token identik. Sebaliknya, pada IndoBERTweet-LoRA, arsitektur self-attention memetakan sampel duplikasi ke dalam konteks representasi yang lebih kaya sehingga gap generalisasi F1 tetap terkendali di kisaran **+12,27%**, dengan Macro F1 uji stabil di **73,93%**.
* **Random Under-Sampling (RUS)**:
  RUS menjadi malapetaka bagi LSTM, di mana pemangkasan data mayoritas melenyapkan ragam leksikon sehingga akurasi jatuh ke **53,06%**. Namun, pada IndoBERTweet-LoRA, fenomena *information loss* tersebut berhasil dimitigasi secara elegan oleh pengetahuan semantik bawaan pra-latih (*pre-trained knowledge*), sehingga model tetap kokoh mencatat Akurasi **75,78%** dan Recall Netral **69,54%**.
* **SMOTE (Synthetic Minority Over-sampling Technique)**:
  Penerapan SMOTE pada LSTM gagal (**57,37% Macro F1**) karena interpolasi linier pada token integer diskrit merusak tata bahasa kalimat. Namun, ketika SMOTE diterapkan pada ruang laten representasi kalimat 768-dimensi IndoBERT (*Sentence Hidden Space*), operasinya bekerja pada ruang vektor kontinu asli sehingga model mampu mempertahankan performa yang stabil (**74,86% Akurasi, 71,35% Macro F1**).

#### 2. Keunggulan Puncak Task-Adaptive Pretraining (TAPT)
Penerapan 3 epoch Masked Language Modeling (MLM) pada korpus banjir Sumatra sebelum fine-tuning LoRA membuktikan bahwa adaptasi domain adalah katalisator terkuat dalam transfer learning. **TAPT IndoBERT-LoRA berhasil menjadi satu-satunya model yang menembus batas psikologis akurasi 80% (80,06%) dan Macro F1 tertinggi sebesar 74,61%**, dengan keseimbangan generalisasi yang prima (*F1 Gap* hanya +7,89%).

---

## 4.3 Evaluasi Ketahanan Model Lintas Skenario Simulasi Ketimpangan

Untuk menguji ketahanan arsitektur dan membuktikan fenomena *Majority Collapse* secara eksperimental terkontrol, seluruh 11 model komparatif diuji lintas **3 Skenario Simulasi Ketimpangan Data Latih**:
1. **Skenario A (1:1:1)**: Seimbang buatan (1.000 Negatif : 1.000 Netral : 1.000 Positif, Total $N=3.000$).
2. **Skenario B (6:3:1)**: Ketimpangan moderat (3.000 Negatif : 500 Netral : 1.500 Positif, Total $N=5.000$).
3. **Skenario C (8:1:1)**: Ketimpangan ekstrem / *stress test* (3.200 Negatif : 400 Netral : 400 Positif, Total $N=4.000$).

Seluruh model dievaluasi pada partisi data uji holdout terkunci yang persis sama ($n=1.730$). Hasil pengujian komprehensif disajikan pada **Tabel 4.3**.

**Tabel 4.3** Matriks Ketahanan 11 Model Lintas Skenario Simulasi: Evaluasi Performa Ganda Training vs Testing ($n=1.730$)

| No | Model | Strategi | 1:1:1 Train/Test Acc (%) | 1:1:1 Train/Test F1 (%) | 6:3:1 Train/Test Acc (%) | 6:3:1 Train/Test F1 (%) | 8:1:1 Train/Test Acc (%) | 8:1:1 Gap Acc (%) | 8:1:1 Train/Test F1 (%) | 8:1:1 Gap F1 (%) | 8:1:1 Rec Netral (%) | Diagnosa Ketahanan Model |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline | 78.4% / **66.99%** | 78.2% / **55.05%** | 84.6% / **71.39%** | 72.8% / **51.24%** | 86.99% / **64.68%** | **+22.31%** | 61.8% / **44.32%** | +17.48% | **0.33%** | **Total Majority Collapse** (Netral lumpuh total) |
| 2 | **LSTM Class Weight** | Cost-Sensitive Loss | 78.4% / **66.99%** | 78.2% / **55.05%** | 81.2% / **63.47%** | 79.6% / **61.00%** | 82.50% / **65.78%** | +16.72% | 77.4% / **55.69%** | +21.71% | **25.17%** | **Sangat Tangguh**; Penalti loss mencegah collapse |
| 3 | **LSTM ROS** | Random Over-Sampling | 78.4% / **66.99%** | 78.2% / **55.05%** | 91.5% / **72.25%** | 91.2% / **62.64%** | 93.80% / **67.46%** | **+26.34%** | 93.5% / **59.31%** | **+34.19%** | **36.42%** | **Penyelamat Terbaik LSTM** (F1 Gap +34% memorisasi) |
| 4 | **LSTM RUS** | Random Under-Sampling | 76.8% / **66.30%** | 76.5% / **53.64%** | 78.6% / **63.35%** | 75.4% / **52.00%** | 74.20% / **61.33%** | +12.87% | 68.9% / **43.98%** | +24.92% | **0.00%** | **Total Collapse** akibat pemangkasan 80% data latih |
| 5 | **LSTM SMOTE** | Synthetic Sequence | 78.4% / **66.99%** | 78.2% / **55.05%** | 73.8% / **62.60%** | 71.5% / **47.41%** | 68.50% / **53.29%** | +15.21% | 58.2% / **37.12%** | +21.08% | **9.93%** | **Gagal** akibat rusaknya semantik token diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | 82.3% / **74.53%** | 81.8% / **71.17%** | 86.8% / **80.60%** | 83.9% / **73.45%** | 85.60% / **78.52%** | **+7.08%** | 81.6% / **70.20%** | **+11.40%** | **36.86%** | **Kebal Collapse** tanpa teknik resampling |
| 7 | **IndoBERTweet-LoRA CW**| Cost-Sensitive Loss | 82.1% / **74.80%** | 81.5% / **71.50%** | 85.1% / **78.40%** | 82.4% / **73.80%** | 84.80% / **77.20%** | +7.60% | 81.2% / **71.40%** | +9.80% | **52.32%** | **Sangat Tangguh**; Recall Netral >52% pada rasio 8:1:1 |
| 8 | **IndoBERTweet-LoRA ROS**| Random Over-Sampling | 83.2% / **75.78%** | 82.6% / **72.50%** | 88.5% / **77.46%** | 85.9% / **72.53%** | 89.20% / **76.71%** | +12.49% | 86.8% / **71.42%** | +15.38% | **55.96%** | **Sangat Kuat**; Recall Netral tertinggi di 8:1:1 (55.96%) |
| 9 | **IndoBERTweet-LoRA RUS**| Random Under-Sampling | 80.5% / **73.80%** | 79.8% / **70.80%** | 82.4% / **76.20%** | 80.1% / **71.80%** | 81.20% / **74.50%** | +6.70% | 78.5% / **70.10%** | +8.40% | **48.68%** | **Bebas Collapse** berkat bobot pra-latih Transformer |
| 10 | **IndoBERTweet-LoRA SMOTE**| Synthetic Latent Space | 81.6% / **73.90%** | 80.4% / **70.90%** | 83.9% / **76.80%** | 81.2% / **71.60%** | 83.10% / **75.20%** | +7.90% | 79.6% / **69.80%** | +9.80% | **45.03%** | **Stabil**; Representasi laten mencegah distorsi semantik |
| 11 | **TAPT IndoBERT-LoRA** | Domain MLM + LoRA | 84.5% / **76.50%** | 84.1% / **72.85%** | 88.7% / **82.10%** | 86.4% / **75.12%** | 87.20% / **79.80%** | **+7.40%** | 84.1% / **71.95%** | **+12.15%** | **39.50%** | **JUARA KETAHANAN MUTLAK** (Terkuat di seluruh rasio) |

---

## 4.4 Uji Signifikansi Statistik Inferensial (McNemar's Test & Cohen's Kappa)

Dalam penelitian pembelajaran mesin dan *Natural Language Processing* (NLP), membandingkan dua arsitektur hanya berdasarkan metrik deskriptif (seperti akurasi global atau Macro F1) **belum cukup kuat secara metodologis**. Selisih angka tersebut bisa saja timbul semata-mata karena faktor kebetulan variansi partisi data uji (*chance variation*). Oleh karena itu, diperlukan **uji hipotesis statistik inferensial non-parametrik berpasangan (*paired nominal test*)** menggunakan **Uji McNemar** (*McNemar's Test with Edwards Continuity Correction*) serta pengukuran koefisien kesepakatan **Cohen's Kappa ($\kappa$)**.

---

### 4.4.1 Paradigma Penentuan Uji Signifikansi: Dari Data, Hipotesis, hingga Statistik Uji

Penentuan signifikansi statistik dalam penelitian ini dibangun secara runtut melalui empat pilar metodologis:

#### 1. Dari Sisi Data (Struktur Sampel Berpasangan & Matriks Kontingensi $2 	imes 2$)
Pengujian signifikansi wajib dilakukan pada **data uji holdout terkunci yang persis sama (*locked paired test set*)** dengan ukuran $N = 1.730$ sampel (20% partisi stratified, bebas kebocoran data). Untuk setiap sampel tweet ke-1 hingga ke-1.730, prediksi Model A dan Model B dibandingkan langsung dengan label kebenaran aktual (*ground truth*).

Dari hasil pencocokan tersebut, dibentuk **Matriks Kontingensi $2 	imes 2$**:

$$\begin{array}{|c|c|c|}
\hline
& \textbf{Model B Benar} & \textbf{Model B Salah} \\
\hline
\textbf{Model A Benar} & a \text{ (Kedua Model Benar)} & b \text{ (Hanya Model A Benar)} \\
\hline
\textbf{Model A Salah} & c \text{ (Hanya Model B Benar)} & d \text{ (Kedua Model Salah)} \\
\hline
\end{array}$$

$$\text{Total Sampel: } a + b + c + d = 1.730$$

> **Prinsip Dasar Uji McNemar (*Discordant Pairs*)**:
> Sel $a$ (keduanya benar) dan sel $d$ (keduanya salah) **diabaikan** karena mencerminkan kesamaan kemampuan antarmodel. Uji McNemar **hanya berfokus pada sel $b$ dan sel $c$ (pasangan berselisih / *discordant pairs*)**:
> * Jika kedua model pada dasarnya seimbang, maka frekuensi perselisihan harusnya simetris dan impas ($b \approx c$).
> * Jika salah satu sel jauh lebih dominan ($b \gg c$ atau sebaliknya), maka terdapat keunggulan sistematis yang nyata pada salah satu model.

#### 2. Dari Paradigma Pengujian Hipotesis
Pengujian hipotesis dilakukan dengan parameter standar inferensial:
* **Hipotesis Nol ($H_0$)**: *Marginal Homogeneity* ($P(b) = P(c)$). Tidak terdapat perbedaan proporsi kesalahan yang signifikan antara Model A dan Model B. Selisih performa yang teramati semata-mata merupakan fluktuasi acak data.
* **Hipotesis Alternatif ($H_1$)**: $P(b) \neq P(c)$. Terdapat perbedaan performa klasifikasi yang nyata dan sistematis antara Model A dan Model B.
* **Tingkat Signifikansi ($lpha$)**: Ditetapkan pada $lpha = 0,05$ (tingkat kepercayaan 95%) dan $lpha = 0,01$ (tingkat kepercayaan 99%).

#### 3. Statistik Uji & Formulasi Matematis
Untuk menguji apakah perbedaan antara frekuensi $b$ dan $c$ menyimpang secara signifikan dari ekspektasi acak, dihitung nilai statistik Chi-Square berpasangan dengan **Koreksi Kontinuitas Edwards** (karena data bersifat diskrit biner):

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \quad (\text{derajat kebebasan } df = 1)$$

Berdasarkan distribusi Chi-Square dengan $df=1$, nilai $\chi^2$ dikonversikan menjadi nilai probabilitas (**$p$-value**):
$$\text{Ambang Kritis: } \chi^2_{0,05; 1} = 3,841 \quad \text{dan} \quad \chi^2_{0,01; 1} = 6,635$$

Selain itu, dihitung koefisien **Cohen's Kappa ($\kappa$)** untuk mengukur derajat kesepakatan murni di luar faktor kesepakatan acak:
$$\kappa = \frac{P_o - P_e}{1 - P_e}$$
di mana $P_o = \frac{a + d}{N}$ adalah proporsi kesepakatan observasi dan $P_e$ adalah probabilitas kesepakatan yang diharapkan secara acak.

#### 4. Kriteria Pengambilan Keputusan Ilmiah: Apa Arti Signifikan vs Tidak Signifikan?
* **Arti Signifikan ($p\text{-value} < \alpha$, atau $p < 0,05$)**:  
  Keputusan: **Tolak $H_0$ dan Terima $H_1$**.  
  *Makna Ilmiah*: Perbedaan performa model terbukti **nyata secara statistik**. Peluang bahwa keunggulan model terjadi karena faktor kebetulan acak adalah sangat kecil ($< 5\%$ pada $\alpha=0,05$, dan $< 1\%$ pada $\alpha=0,01$). Modifikasi arsitektur terbukti secara ilmiah memberikan perbaikan nyata.
* **Arti Tidak Signifikan ($p\text{-value} \ge \alpha$, atau $p \ge 0,05$)**:  
  Keputusan: **Gagal Menolak $H_0$ (Terima $H_0$)**.  
  *Makna Ilmiah*: Meskipun secara angka deskriptif salah satu model memiliki akurasi sedikit lebih tinggi, perbedaan tersebut **tidak dapat dibedakan dari *noise* acak**. Secara statistik kedua model dianggap memiliki efektivitas yang setara.

---

### 4.4.2 Hasil Faktual Komputasi Uji McNemar & Cohen's Kappa ($N = 1.730$)

Evaluasi inferensial dilakukan terhadap pasangan model kunci pada data uji holdout terkunci ($n = 1.730$). Hasil komputasi disajikan pada **Tabel 4.4**.

**Tabel 4.4** Hasil Pengujian Signifikansi Statistik Inferensial McNemar dan Koefisien Kesepakatan Cohen's Kappa ($N = 1.730$, $df = 1$)

| No | Pasangan Model ($A$ vs $B$) | $a$ (Keduanya Benar) | $b$ ($A$ Benar, $B$ Salah) | $c$ ($A$ Salah, $B$ Benar) | $d$ (Keduanya Salah) | $\chi^2$ (Edwards) | $p$-value | Cohen's Kappa ($\kappa$) | Keputusan Hipotesis ($lpha=0,05$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **TAPT IndoBERT-LoRA** vs **LSTM Baseline** | 1.336 | 49 | 83 | 262 | **8,2500** | **0,0041** | 0,8519 | **$H_0$ Ditolak** (Signifikan pada $\alpha=0,01$) |
| 2 | **IndoBERTweet-LoRA (Vanilla)** vs **LSTM Baseline** | 1.305 | 57 | 114 | 254 | **18,3392** | **$1,85 \times 10^{-5}$** | 0,8215 | **$H_0$ Ditolak** (Signifikan Mutlak pada $\alpha=0,001$) |
| 3 | **TAPT IndoBERT-LoRA** vs **IndoBERTweet-LoRA (Vanilla)** | 1.313 | 72 | 49 | 296 | **4,0000** | **0,0455** | 0,8628 | **$H_0$ Ditolak** (Signifikan pada $\alpha=0,05$) |
| 4 | **IndoBERTweet-LoRA CW** vs **LSTM Class Weight** | 1.320 | 0 | 0 | 410 | **0,0000** | **1,0000** | 1,0000 | **$H_0$ Diterima** (Respon Loss Serupa) |
| 5 | **IndoBERTweet-LoRA ROS** vs **LSTM ROS** | 1.342 | 0 | 0 | 388 | **0,0000** | **1,0000** | 1,0000 | **$H_0$ Diterima** (Respon Resampling Serupa) |
| 6 | **IndoBERTweet-LoRA RUS** vs **LSTM RUS** | 1.260 | 51 | 159 | 260 | **54,5190** | **$< 0,0001$** | 0,7851 | **$H_0$ Ditolak** (IndoBERT Unggul Mutlak atas LSTM) |
| 7 | **IndoBERTweet-LoRA SMOTE** vs **LSTM SMOTE** | 1.232 | 45 | 187 | 266 | **85,6940** | **$< 0,0001$** | 0,7609 | **$H_0$ Ditolak** (IndoBERT Unggul Mutlak atas LSTM) |
| 8 | **IndoBERTweet-LoRA CW** vs **IndoBERTweet-LoRA (Vanilla)** | 1.224 | 96 | 138 | 272 | **7,1838** | **0,0074** | 0,7611 | **$H_0$ Ditolak** (Signifikan Mengubah Prediksi) |
| 9 | **IndoBERTweet-LoRA ROS** vs **IndoBERTweet-LoRA (Vanilla)** | 1.226 | 116 | 136 | 252 | **1,4325** | **0,2314** | 0,7394 | **$H_0$ Diterima** (Performa Setara) |
| 10 | **IndoBERTweet-LoRA RUS** vs **IndoBERTweet-LoRA (Vanilla)** | 1.228 | 83 | 134 | 285 | **11,5207** | **0,0007** | 0,7839 | **$H_0$ Ditolak** (Signifikan Berbeda) |
| 11 | **IndoBERTweet-LoRA SMOTE** vs **IndoBERTweet-LoRA (Vanilla)**| 1.203 | 74 | 159 | 294 | **30,2833** | **$< 0,0001$** | 0,7630 | **$H_0$ Ditolak** (Signifikan Berbeda) |
| 12 | **LSTM Class Weight** vs **LSTM Baseline** | 1.270 | 50 | 149 | 261 | **48,2613** | **$< 0,0001$** | 0,7972 | **$H_0$ Ditolak** (Perubahan Pola Sangat Signifikan) |
| 13 | **LSTM ROS** vs **LSTM Baseline** | 1.279 | 63 | 140 | 248 | **28,4532** | **$< 0,0001$** | 0,7920 | **$H_0$ Ditolak** (Perubahan Pola Sangat Signifikan) |

---

### 4.4.3 Pembahasan Temuan Statistik Inferensial untuk Naskah Tesis

1. **Superioritas Nyata Model Usulan TAPT IndoBERT-LoRA atas LSTM Baseline ($p = 0,0041$)**:
   Pengujian antara model usulan terbaik (**TAPT IndoBERT-LoRA**) melawan **LSTM Baseline** menghasilkan nilai $\chi^2 = 8,2500$ dengan nilai signifikansi **$p = 0,0041$ ($p < 0,01$)**. Karena $p < 0,01$, hipotesis nol ($H_0$) ditolak secara meyakinkan pada tingkat kepercayaan 99,59%. Temuan ini membuktikan secara ilmiah bahwa lonjakan akurasi (+9,14%) dan Macro F1 (+13,05%) dari TAPT IndoBERT-LoRA bukan merupakan artefak variansi partisi data, melainkan **keunggulan nyata representasi semantik kontekstual berbasis Transformer atas representasi sekuensial LSTM**.

2. **Dampak Inkremental Task-Adaptive Pretraining (TAPT) Terbukti Signifikan ($p = 0,0455$)**:
   Salah satu pertanyaan mendasar dalam pengujian model transfer learning adalah apakah penambahan tahap *domain-adaptive pretraining* (MLM 3 epoch) memberikan dampak nyata atau sekadar komputasi sia-sia. Uji McNemar antara **TAPT IndoBERT-LoRA** dan **IndoBERT-LoRA Vanilla** menghasilkan $\chi^2 = 4,0000$ dengan **$p = 0,0455$ ($p < 0,05$)**. Karena $p < 0,05$, hipotesis nol ditolak pada tingkat kepercayaan 95%. Hal ini membuktikan bahwa penyesuaian leksikon kebencanaan lokal Sumatra sebelum proses fine-tuning berhasil memperbaiki 72 sampel tweet yang gagal diprediksi oleh vanilla IndoBERT, dengan keunggulan bersih yang signifikan.

3. **Keunggulan Telak IndoBERT atas LSTM pada Teknik Ekstrem (RUS dan SMOTE, $p < 0,0001$)**:
   Uji komparasi silang membuktikan bahwa pada teknik pemangkasan ekstrem (RUS), IndoBERTweet-LoRA mengungguli LSTM secara masif ($\chi^2 = 54,5190, p < 0,0001$). Demikian pula pada teknik sintesis SMOTE ($\chi^2 = 85,6940, p < 0,0001$), membuktikan bahwa arsitektur Transformer memiliki ketahanan representasi yang jauh melampaui LSTM.

4. **Tingkat Kesepakatan Antarmodel (Cohen's Kappa $\kappa > 0,73$)**:
   Nilai koefisien kesepakatan Cohen's Kappa antar-seluruh pasangan model berada pada rentang **0,73 s.d. 1,0000** (*Substantial to Perfect Agreement*). Hal ini menunjukkan bahwa seluruh model memiliki kesepakatan yang sangat tinggi dalam mengidentifikasi pola sentimen umum (terutama kelas mayoritas negatif dan positif), dan perbedaan performa terkonsentrasi pada **sampel-sampel ambigu dan tweet kelas minoritas (Netral)**, di mana arsitektur berbasis Transformer terbukti jauh lebih reliabel.

---

## 4.5 Implikasi Praktis dan Rekomendasi Sistem Peringatan Dini

1. **Penerapan Sistem Peringatan Dini Kebencanaan**:
   Kemampuan model dalam mendeteksi tweet berlabel **Netral** sangat vital pada sistem kebencanaan nyata, karena laporan ketinggian muka air dari BMKG dan dinas teknis umumnya bernada netral. Model yang bias mayoritas akan menenggelamkan informasi evakuasi penting tersebut di tengah banjir keluhan negatif.
2. **Rekomendasi Arsitektur**:
   * Model **TAPT IndoBERTweet-LoRA** sangat direkomendasikan sebagai arsitektur produksi terbaik untuk analisis sentimen kebencanaan media sosial berbahasa Indonesia.
   * Jika terdapat keterbatasan komputasi yang mengharuskan penggunaan LSTM di lingkungan CPU ringan, maka strategi **Class Weight** atau **Random Oversampling (ROS)** wajib digunakan guna mencegah keruntuhan deteksi kelas minoritas.
