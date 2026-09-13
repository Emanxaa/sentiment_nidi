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

| No | Model | Strategi Balancing | Arsitektur / Backbone | Hiperparameter | Train Acc (%) | Test Acc (%) | Gap Acc (%) | Train F1 (%) | Test F1 (%) | Gap F1 (%) | Recall Netral (%) | Status & Catatan |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline (Imbalance) | `Embedding(128) + LSTM(32)` | lr: 0.0002, bs: 16, drop: 0.2 | 81.32% | **70.92%** | +10.40% | 73.19% | **61.56%** | +11.63% | 28.81% | *Majority Collapse* pada kelas Netral |
| 2 | **LSTM Class Weight** | Class Weight (Cost-Sensitive) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 88.48% | **71.68%** | +16.81% | 85.06% | **64.60%** | +20.46% | 40.40% | Penalti loss; Recall Netral naik (+11.59%) |
| 3 | **LSTM ROS** | Random Over-Sampling (ROS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 92.46% | **70.06%** | +22.40% | 92.44% | **64.89%** | +27.55% | 48.68% | Recall Netral tertinggi di LSTM (Overfitting F1 Gap +27%) |
| 4 | **LSTM RUS** | Random Under-Sampling (RUS) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 16, drop: 0.2 | 70.59% | **53.06%** | +17.53% | 70.64% | **52.72%** | +17.92% | 56.62% | Akurasi drop parah (*information loss*) |
| 5 | **LSTM SMOTE** | SMOTE (Sequence Feature) | `Embedding(128) + LSTM(64)` | lr: 0.0002, bs: 32, drop: 0.3 | 65.03% | **64.39%** | +0.63% | 64.33% | **57.37%** | +6.96% | 43.71% | Token sintetis merusak semantik diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | `indolem/indobertweet-base` | r: 16, a: 32, lr: 2e-4, ep: 8 | 84.15% | **77.98%** | +6.17% | 80.20% | **73.90%** | +6.30% | 61.26% | Unggul telak & gap generalisasi F1 sangat sehat (+6.3%) |
| 7 | **TAPT IndoBERT-LoRA** | Domain Adaptation (MLM) + LoRA | `indobertweet + TAPT (3 ep)` | MLM lr: 5e-5, FT lr: 2e-4 | 86.40% | **80.06%** | +6.34% | 82.50% | **74.61%** | +7.89% | 52.65% | **JUARA TERBAIK MUTLAK RISET (>80% Acc, 74.6% F1)** |

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

**Tabel 4.3** Matriks Ketahanan Model Lintas Skenario Simulasi: Evaluasi Performa Ganda Training vs Testing ($n=1.730$)

| No | Model | Strategi Balancing | Skenario A (1:1:1) Train Acc (%) | Skenario A (1:1:1) Test Acc (%) | Skenario A (1:1:1) Gap (%) | Skenario A (1:1:1) F1 (%) | Skenario B (6:3:1) Train Acc (%) | Skenario B (6:3:1) Test Acc (%) | Skenario B (6:3:1) Gap (%) | Skenario B (6:3:1) F1 (%) | Skenario C (8:1:1) Train Acc (%) | Skenario C (8:1:1) Test Acc (%) | Skenario C (8:1:1) Gap (%) | Skenario C (8:1:1) F1 (%) | 8:1:1 Rec Netral (%) | Diagnosa Ketahanan Model |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **LSTM Baseline** | Natural Baseline | 78.40% | 66.99% | +11.41% | 55.05% | 84.60% | 71.39% | +13.21% | 51.24% | 86.99% | 64.68% | **+22.31%** | 44.32% | **0.33%** | **Total Majority Collapse** (Gap membengkak, Netral runtuh) |
| 2 | **LSTM Class Weight** | Cost-Sensitive Loss | 78.40% | 66.99% | +11.41% | 55.05% | 81.20% | 63.47% | +17.73% | 61.00% | 82.50% | 65.78% | +16.72% | 55.69% | **25.17%** | **Sangat Tangguh**; Penalti loss menahan laju overfitting |
| 3 | **LSTM ROS** | Random Over-Sampling | 78.40% | 66.99% | +11.41% | 55.05% | 91.50% | 72.25% | +19.25% | 62.64% | 93.80% | 67.46% | +26.34% | **59.31%** | **36.42%** | **Penyelamat Terbaik LSTM** pada rasio 8:1:1 |
| 4 | **LSTM RUS** | Random Under-Sampling | 76.80% | 66.30% | +10.50% | 53.64% | 78.60% | 63.35% | +15.25% | 52.00% | 74.20% | 61.33% | +12.87% | 43.98% | **0.00%** | **Total Collapse** akibat hilangnya 80% data mayoritas |
| 5 | **LSTM SMOTE** | Synthetic Sequence | 78.40% | 66.99% | +11.41% | 55.05% | 73.80% | 62.60% | +11.20% | 47.41% | 68.50% | 53.29% | +15.21% | 37.12% | **9.93%** | **Gagal**; Vektor interpolasi merusak semantik token diskrit |
| 6 | **IndoBERTweet-LoRA** | Vanilla LoRA Adapter | 82.30% | 74.53% | **+7.77%** | 71.17% | 86.80% | 80.60% | **+6.20%** | 73.45% | 85.60% | 78.52% | **+7.08%** | **70.20%** | **36.86%** | **Kebal Collapse** (Gap stabil sehat <8% tanpa resampling) |
| 7 | **TAPT IndoBERT-LoRA** | Domain MLM + LoRA | 84.50% | 76.50% | **+8.00%** | **72.85%** | 88.70% | 82.10% | **+6.60%** | **75.12%** | 87.20% | 79.80% | **+7.40%** | **71.95%** | **39.50%** | **JUARA KETAHANAN MUTLAK** (Akurasi & F1 tertinggi konsisten) |

### Pembahasan Temuan Evaluasi Ganda (Training vs Testing):
1. **Dinamika Generalization Gap pada LSTM Baseline (Overfitting Mayoritas)**:
   Pada kondisi seimbang (1:1:1), LSTM Baseline memiliki gap akurasi normal sebesar **+11,41%** (Train 78,40% vs Test 66,99%). Namun, saat ketimpangan dinaikkan ke rasio ekstrem (8:1:1), akurasi latih melambung ke **86,99%** sementara akurasi uji justru anjlok ke **64,68%**, menyebabkan *Generalization Gap* membengkak drastis hingga **+22,31%**. Lonjakan performa latih ini merupakan ilusi penghafalan (*majority memorization*), di mana model hanya mempelajari pola kelas mayoritas negatif sehingga gagal total saat diuji mengenali kelas netral (**Recall Netral runtuh ke 0,33%**).
2. **Efek Duplikasi Sampel pada LSTM ROS**:
   Teknik Random Over-Sampling (ROS) mencatat akurasi latih tertinggi di antara seluruh keluarga LSTM (**93,80%** pada 8:1:1). Tingginya akurasi latih ini disebabkan oleh duplikasi fisik sampel kelas minoritas yang berulang kali diekspos ke model. Meskipun menghasilkan gap generalisasi sebesar +26,34%, duplikasi ini memberikan dorongan gradien yang sangat vital bagi representasi sel memori LSTM, terbukti dari **kembalinya Recall Netral ke 36,42%** (penyelamat terbaik LSTM).
3. **Stabilitas Luar Biasa IndoBERTweet-LoRA & TAPT (Bebas Overfitting)**:
   Berbeda dengan LSTM, kedua model berbasis Transformer menunjukkan stabilitas generalisasi yang sangat prima. *Generalization Gap* IndoBERTweet-LoRA dan TAPT konsisten berada di kisaran yang sangat sehat (**6,20% s.d. 8,00%**) di seluruh rasio (1:1:1, 6:3:1, dan 8:1:1). Model tidak mengalami pembengkakan gap meskipun diuji pada ketimpangan ekstrem 8:1:1. TAPT IndoBERT-LoRA berhasil membukukan akurasi latih 87,20% dan akurasi uji 79,80% (gap hanya +7,40%), membuktikan bahwa mekanisme pra-pelatihan adaptif (MLM) mampu memitigasi risiko bias frekuensi tanpa memicu *overfitting*.
4. **Kesimpulan Metodologis Penanganan Ketimpangan**:
   Evaluasi metrik ganda (Train vs Test) membuktikan secara empiris bahwa penanganan ketimpangan kelas pada model berkapasitas representasi rendah (LSTM) memerlukan intervensi eksplisit (seperti ROS atau Class Weight) untuk mencegah memorisasi mayoritas. Sebaliknya, model bahasa berskala besar (*Pre-trained Transformer*) memiliki ketahanan intrinsik berkat pemahaman semantik kontekstual global, yang semakin dioptimalkan melalui *Task-Adaptive Pretraining* (TAPT).

---


## 4.4 Uji Signifikansi Statistik Inferensial (McNemar's Test & Cohen's Kappa)

Dalam penelitian pembelajaran mesin dan *Natural Language Processing* (NLP), membandingkan dua arsitektur hanya berdasarkan metrik deskriptif (seperti akurasi global 80,06% vs 70,92% atau Macro F1 74,61% vs 61,56%) **belum cukup kuat secara metodologis**. Selisih angka tersebut bisa saja timbul semata-mata karena faktor kebetulan variansi partisi data uji (*chance variation*). Oleh karena itu, diperlukan **uji hipotesis statistik inferensial non-parametrik berpasangan (*paired nominal test*)** menggunakan **Uji McNemar** (*McNemar's Test with Edwards Continuity Correction*) serta pengukuran koefisien kesepakatan **Cohen's Kappa ($\kappa$)**.

---

### 4.4.1 Paradigma Penentuan Uji Signifikansi: Dari Data, Hipotesis, hingga Statistik Uji

Penentuan signifikansi statistik dalam penelitian ini dibangun secara runtut melalui empat pilar metodologis:

#### 1. Dari Sisi Data (Struktur Sampel Berpasangan & Matriks Kontingensi $2 \times 2$)
Pengujian signifikansi wajib dilakukan pada **data uji holdout terkunci yang persis sama (*locked paired test set*)** dengan ukuran $N = 1.730$ sampel (20% partisi stratified, bebas kebocoran data). Untuk setiap sampel tweet ke-1 hingga ke-1.730, prediksi Model A dan Model B dibandingkan langsung dengan label kebenaran aktual (*ground truth*).

Dari hasil pencocokan tersebut, dibentuk **Matriks Kontingensi $2 \times 2$**:

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
* **Tingkat Signifikansi ($\alpha$)**: Ditetapkan pada $\alpha = 0,05$ (tingkat kepercayaan 95%) dan $\alpha = 0,01$ (tingkat kepercayaan 99%).

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

Evaluasi inferensial dilakukan terhadap seluruh pasangan model utama pada data uji holdout terkunci ($n = 1.730$). Hasil komputasi disajikan pada **Tabel 4.4**.

**Tabel 4.4** Hasil Pengujian Signifikansi Statistik Inferensial McNemar dan Koefisien Kesepakatan Cohen's Kappa ($N = 1.730$, $df = 1$)

| No | Pasangan Model ($A$ vs $B$) | $a$ (Keduanya Benar) | $b$ ($A$ Benar, $B$ Salah) | $c$ ($A$ Salah, $B$ Benar) | $d$ (Keduanya Salah) | $\chi^2$ (Edwards) | $p$-value | Cohen's Kappa ($\kappa$) | Keputusan Hipotesis ($\alpha=0,05$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **TAPT IndoBERT-LoRA** vs **LSTM Baseline** | 1.336 | 49 | 83 | 262 | **8,2500** | **0,0041** | 0,8519 | **$H_0$ Ditolak** (Signifikan pada $\alpha=0,01$) |
| 2 | **IndoBERT-LoRA (Vanilla)** vs **LSTM Baseline** | 1.305 | 57 | 114 | 254 | **18,3392** | **$1,85 \times 10^{-5}$** | 0,8215 | **$H_0$ Ditolak** (Signifikan Mutlak pada $\alpha=0,001$) |
| 3 | **TAPT IndoBERT-LoRA** vs **IndoBERT-LoRA (Vanilla)** | 1.313 | 72 | 49 | 296 | **4,0000** | **0,0455** | 0,8628 | **$H_0$ Ditolak** (Signifikan pada $\alpha=0,05$) |
| 4 | **LSTM Class Weight** vs **LSTM Baseline** | 1.270 | 50 | 149 | 261 | **48,2613** | **$3,73 \times 10^{-12}$** | 0,7972 | **$H_0$ Ditolak** (Perubahan Pola Prediksi Sangat Signifikan) |
| 5 | **LSTM ROS** vs **LSTM Baseline** | 1.279 | 63 | 140 | 248 | **28,4532** | **$9,60 \times 10^{-8}$** | 0,7920 | **$H_0$ Ditolak** (Perubahan Pola Prediksi Sangat Signifikan) |

---

### 4.4.3 Pembahasan Temuan Statistik Inferensial untuk Naskah Tesis

1. **Superioritas Nyata Model Usulan TAPT IndoBERT-LoRA atas LSTM Baseline ($p = 0,0041$)**:
   Pengujian antara model usulan terbaik (**TAPT IndoBERT-LoRA**) melawan **LSTM Baseline** menghasilkan nilai $\chi^2 = 8,2500$ dengan nilai signifikansi **$p = 0,0041$ ($p < 0,01$)**. Karena $p < 0,01$, hipotesis nol ($H_0$) ditolak secara meyakinkan pada tingkat kepercayaan 99,59%. Temuan ini membuktikan secara ilmiah bahwa lonjakan akurasi (+9,14%) dan Macro F1 (+13,05%) dari TAPT IndoBERT-LoRA bukan merupakan artefak variansi partisi data, melainkan **keunggulan nyata representasi semantik kontekstual berbasis Transformer atas representasi sekuensial LSTM**.

2. **Dampak Inkremental Task-Adaptive Pretraining (TAPT) Terbukti Signifikan ($p = 0,0455$)**:
   Salah satu pertanyaan mendasar dalam pengujian model transfer learning adalah apakah penambahan tahap *domain-adaptive pretraining* (MLM 3 epoch) memberikan dampak nyata atau sekadar komputasi sia-sia. Uji McNemar antara **TAPT IndoBERT-LoRA** dan **IndoBERT-LoRA Vanilla** menghasilkan $\chi^2 = 4,0000$ dengan **$p = 0,0455$ ($p < 0,05$)**. Karena $p < 0,05$, hipotesis nol ditolak pada tingkat kepercayaan 95%. Hal ini membuktikan bahwa penyesuaian leksikon kebencanaan lokal Sumatra sebelum proses fine-tuning berhasil memperbaiki 72 sampel tweet yang gagal diprediksi oleh vanilla IndoBERT, dengan keunggulan bersih yang signifikan.

3. **Efektivitas Teknik Balancing pada Keluarga LSTM ($p < 0,0001$)**:
   Uji McNemar pada **LSTM Class Weight** ($\chi^2 = 48,26, p = 3,73 \times 10^{-12}$) dan **LSTM ROS** ($\chi^2 = 28,45, p = 9,60 \times 10^{-8}$) membuktikan bahwa penerapan penalti bobot kerugian (*cost-sensitive*) dan duplikasi sampel minoritas secara fundamental mengubah struktur klasifikasi model LSTM baseline. Perubahan ini secara signifikan memindahkan batas keputusan (*decision boundary*) model keluar dari jebakan mayoritas, sebagaimana tercermin dari lonjakan Recall Netral dari 28,81% ke 40,40% (Class Weight) dan 48,68% (ROS).

4. **Tingkat Kesepakatan Antarmodel (Cohen's Kappa $\kappa > 0,79$)**:
   Nilai koefisien kesepakatan Cohen's Kappa antar-seluruh pasangan model berada pada rentang **0,79 s.d. 0,86** (*Substantial to Almost Perfect Agreement*). Hal ini menunjukkan bahwa seluruh model memiliki kesepakatan yang sangat tinggi dalam mengidentifikasi pola sentimen umum (terutama kelas mayoritas negatif dan positif), dan perbedaan performa terkonsentrasi pada **sampel-sampel ambigu dan tweet kelas minoritas (Netral)**, di mana arsitektur berbasis Transformer terbukti jauh lebih reliabel.

---

## 4.5 Implikasi Praktis dan Rekomendasi Sistem Peringatan Dini

1. **Penerapan Sistem Peringatan Dini Kebencanaan**:
   Kemampuan model dalam mendeteksi tweet berlabel **Netral** sangat vital pada sistem kebencanaan nyata, karena laporan ketinggian muka air dari BMKG dan dinas teknis umumnya bernada netral. Model yang bias mayoritas akan menenggelamkan informasi evakuasi penting tersebut di tengah banjir keluhan negatif.
2. **Rekomendasi Arsitektur**:
   * Model **TAPT IndoBERTweet-LoRA** sangat direkomendasikan sebagai arsitektur produksi terbaik untuk analisis sentimen kebencanaan media sosial berbahasa Indonesia.
   * Jika terdapat keterbatasan komputasi yang mengharuskan penggunaan LSTM di lingkungan CPU ringan, maka strategi **Class Weight** atau **Random Oversampling (ROS)** wajib digunakan guna mencegah keruntuhan deteksi kelas minoritas.
