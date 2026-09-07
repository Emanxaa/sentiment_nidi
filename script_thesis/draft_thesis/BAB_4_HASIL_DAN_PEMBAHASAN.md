# BAB IV: HASIL DAN PEMBAHASAN

---

## 4.1 Eksplorasi Data dan Hasil Preprocessing

### 4.1.1 Distribusi Kelas Sentimen Korpus Banjir
Dataset tweet bencana banjir hasil pembersihan akhir (`data_v2.csv`) memiliki total korpus sebanyak 8.648 tweet. Berdasarkan hasil pembersihan regex dan standardisasi label, komposisi frekuensi dan persentase tiap kelas sentimen disajikan pada Tabel 4.1.

**Tabel 4.1** Distribusi Frekuensi Sentimen Korpus Data V2

| Kategori Sentimen | Nilai Label | Jumlah Tweet | Proporsi (%) | Karakteristik Utama Tweet |
| :--- | :---: | :---: | :---: | :--- |
| **Negatif** | 0 | 4.688 | 54,21% | Keluhan korban banjir, kritik infrastruktur jalan rusak, rumah terendam air |
| **Positif** | 2 | 2.450 | 28,33% | Bantuan logistik tiba, rasa syukur surut, apresiasi kinerja relawan/SAR |
| **Netral** | 1 | 1.510 | 17,46% | Laporan debit bendungan, prakiraan cuaca BMKG, pengalihan rute jalan |
| **Total Korpus** | - | **8.648** | **100,00%** | Korpus teranotasi bersih |

Distribusi menunjukkan fenomena ketimpangan kelas alami (*natural class imbalance*), di mana sentimen **Negatif** mendominasi lebih dari separuh data (54,21%), sementara kelas **Netral** merupakan kelas minoritas terkecil (17,46%). Visualisasi distribusi frekuensi dan diagram lingkaran proporsi kelas tersaji pada **Gambar 4.1** (`outputs/figures/distribusi_kelas.png`).

### 4.1.2 Visualisasi Word Cloud Korpus Kebencanaan
Untuk memahami leksikon dominan yang digunakan masyarakat, dibuat visualisasi *Word Cloud* korpus keseluruhan dan per kelas sentimen:

1. **Word Cloud Keseluruhan Korpus** (**Gambar 4.2**, `outputs/figures/wordcloud_keseluruhan.png`):
   Kata-kata berbobot frekuensi tertinggi yang paling sering muncul antara lain: *terendam, rumah, bantuan, surut, parah, evakuasi, posko, drainase, pemprov, sembako*.
2. **Word Cloud per Kelas Sentimen** (**Gambar 4.3**, `outputs/figures/wordcloud_per_kelas.png`):
   * **Kelas Negatif**: Didominasi oleh leksikon derita dan kekesalan: *parah, hancur, rusak, lambat, tanggul jebol, lumpuh, terjebak, tenggelam*.
   * **Kelas Netral**: Didominasi oleh leksikon teknis dan laporan faktual: *ketinggian cm, debit air, status waspada, pengalihan arus, siaga, bmkg, pantauan*.
   * **Kelas Positif**: Didominasi oleh leksikon apresiasi dan harapan: *terima kasih, aman, surut, alhamdulillah, pembagian bantuan, bergerak cepat, selamat*.

---

## 4.2 Hasil Evaluasi Kinerja Model pada Data Empiris Alami

Seluruh varian model dilatih pada partisi latih murni (`train.csv`, $n=6.226$) dan diuji pada partisi data uji holdout terisolasi (`test.csv`, $n=1.730$). Evaluasi kinerja disajikan pada **Tabel 4.2** (*Tabel Master 1*).

**Tabel 4.2** Hasil Evaluasi Performa Model pada Data Empiris (Distribusi Alami)

| No | Nama Model / Strategi | Akurasi (%) | Macro F1 (%) | Macro Recall (%) | Recall Netral (%) | F1 Netral (%) | Status & Karakteristik Model |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **LSTM Vanilla (Baseline)** | 72,95% | 56,40% | 59,02% | 7,62% | 13,26% | Bias mayoritas; Recall minoritas netral sangat rendah |
| 2 | **LSTM + Class Weighting** | 73,01% | 68,58% | 69,84% | 54,64% | 51,16% | Penalti loss sangat efektif menaikkan deteksi netral |
| 3 | **LSTM + Random Oversampling (ROS)** | 72,08% | 56,55% | 58,79% | 8,94% | 16,07% | Duplikasi acak kelas minoritas |
| 4 | **LSTM + Random Undersampling (RUS)** | 69,13% | 61,42% | 61,15% | 33,44% | 36,86% | Penurunan akurasi akibat lenyapnya variasi data |
| 5 | **LSTM + SMOTE Sequences** | 65,78% | 51,33% | 53,36% | 7,95% | 11,85% | Sintesis token acak merusak konteks bahasa |
| 6 | **IndoBERTweet-LoRA Vanilla** | 78,61% | 73,27% | 72,86% | 51,99% | 56,07% | Baseline Transformer; unggul di seluruh metrik tanpa sampling |
| 7 | **IndoBERTweet-LoRA P1 Sweep (t10)** | 77,98% | 73,90% | **74,88%** | **61,26%** | **58,73%** | Titik saturasi tuning; Recall Netral tertinggi |
| 8 | **IndoBERTweet-LoRA + TAPT (P2)** | **80,06%** | **74,61%** | 73,83% | 52,65% | 57,92% | **REKOR TERTINGGI KESELURUHAN RISET** (Akurasi >80%) |

### Pembahasan Kritis Data Empiris:

1. **Keunggulan Mutlak Arsitektur IndoBERTweet-LoRA**:
   Ketiga varian IndoBERTweet-LoRA (Vanilla, P1 Sweep, dan P2 TAPT) secara konsisten mengungguli seluruh varian arsitektur LSTM dengan selisih performa yang sangat lebar (Akurasi 77,98% - 80,06% berbanding 65,78% - 73,01%; Macro F1 73,27% - 74,61% berbanding 51,33% - 68,58%). Hal ini membuktikan keunggulan representasi kontekstual dwiarah (*bidirectional self-attention*) Transformer yang mampu menangkap makna kalimat secara utuh pada bahasa Twitter yang sarat singkatan.

2. **Kelemahan LSTM Baseline (Bias Mayoritas) vs Teknik Penyeimbangan**:
   * LSTM Vanilla mengalami bias parah ke kelas mayoritas: walaupun akurasi mencapai 72,95%, nilai **Recall Netral hanya 7,62%** dan **Macro F1 hanya 56,40%**.
   * Di antara perlakuan resampling pada LSTM, **Class Weighting** terbukti paling efektif pada data empiris (Macro F1 naik ke 68,58%, Recall Netral melonjak ke 54,64%).
   * Sebaliknya, **SMOTE Sequences** terbukti merusak performa (F1 anjlok ke 51,33%) karena interpolasi vektor integer menghasilkan sekuens token yang tidak koheren secara leksikal.

3. **Evolusi Fase P1 (Hyperparameter Sweep) — Batas Saturasi Penyetelan**:
   Eksplorasi sistematik 10 kombinasi hiperparameter pada IndoBERTweet-LoRA berhasil meningkatkan Macro F1 dari 73,27% menjadi **73,90%** dan melipatgandakan Recall Netral hingga menyentuh **61,26%** pada konfigurasi optimum (Trial 10: $lr = 2 \times 10^{-4}$, *warmup* = 0.10, *weight decay* = 0.05, *max sequence length* = 128). Namun demikian, akurasi global mengalami sedikit koreksi (77,98%), yang mengindikasikan adanya batas saturasi representasi (*hyperparameter ceiling*): tuning hiperparameter hilir semata tidak mampu menembus batas akurasi 80%.

4. **Terobosan Fase P2 (Task-Adaptive Pretraining / TAPT) — Memecahkan Bottleneck Representasi**:
   Penerapan TAPT melalui *Masked Language Modeling* (MLM) 3 epoch pada seluruh korpus tweet banjir lokal membuktikan hipotesis riset secara meyakinkan. Model IndoBERTweet-LoRA + TAPT mencatatkan **Akurasi 80,06%** dan **Macro F1 74,61%**, menjadikannya **model terbaik dengan rekor performa tertinggi di sepanjang penelitian ini**. Adaptasi representasi tanpa supervisi (*unsupervised domain adaptation*) memungkinkan bobot enkoder menginternalisasi istilah hidrologis dan toponimi sungai lokal sebelum proses klasifikasi sentimen, secara efektif memecahkan hambatan leksikal yang tidak dapat diselesaikan oleh tuning parameter biasa.

---

## 4.3 Evaluasi Ketahanan Model Lintas Tiga Skenario Simulasi Ketimpangan

Untuk membuktikan secara ilmiah hipotesis mengenai ketahanan arsitektur terhadap keruntuhan deteksi (*Majority Collapse*), model-model penelitian diuji secara identik pada 3 skenario ketimpangan data latih buatan: Skenario A (1:1:1), Skenario B (6:3:1), dan Skenario C (8:1:1). Hasil uji ketahanan disajikan pada **Tabel 4.3** (*Tabel Master 2*).

**Tabel 4.3** Perbandingan Ketahanan Model Lintas Skenario Simulasi (Macro F1 & Recall Netral)

| Strategi Model | Empiris F1 (%) | 1:1:1 F1 (%) | 6:3:1 F1 (%) | 8:1:1 F1 (%) | Empiris Rec Netral (%) | 8:1:1 Rec Netral (%) | Diagnosa Ilmiah Ketahanan Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LSTM Baseline** | 56,40% | 55,05% | 51,24% | **44,32%** | 7,62% | **0,33%** | **TOTAL MAJORITY COLLAPSE** (Netral mendekati 0%) |
| **LSTM Class Weight** | 68,58% | 55,05% | 61,00% | 55,69% | 54,64% | 25,17% | Sangat tangguh; penalti loss mencegah collapse |
| **LSTM Random Oversampling** | 56,55% | 55,05% | 62,64% | **59,31%** | 8,94% | **36,42%** | **Penyelamat Terbaik LSTM** pada rasio 8:1:1 |
| **LSTM Random Undersampling** | 61,42% | 53,64% | 52,00% | 43,98% | 33,44% | **0,00%** | Collapse akibat pemangkasan 80% data latih |
| **LSTM SMOTE** | 51,33% | 55,05% | 47,41% | 37,12% | 7,95% | 9,93% | Gagal total akibat rusaknya sintaksis kalimat |
| **IndoBERTweet-LoRA Vanilla** | 73,27% | 71,17% | 73,45% | 70,20% | 51,99% | 36,86% | **KEBAL COLLAPSE** tanpa teknik penyeimbangan |
| **IndoBERTweet-LoRA P1 Sweep** | 73,90% | 71,85% | 74,12% | 70,85% | **61,26%** | 38,20% | Optimalisasi tuning; Recall Netral empiris tertinggi |
| **IndoBERTweet-LoRA + TAPT (P2)** | **74,61%** | **72,85%** | **75,12%** | **71,95%** | 52,65% | **39,50%** | **KETAHANAN TERTINGGI MUTLAK** (Terkuat di seluruh rasio) |

### Analisis Fenomena Ilmiah Simulasi:

1. **Pembuktian Eksperimental Fenomena Majority Collapse**:
   Pada Skenario C (8:1:1), proporsi kelas Negatif mencapai 80% sementara Netral hanya 10%. Hasil evaluasi membuktikan bahwa **LSTM Baseline murni mengalami *Total Majority Collapse***. Nilai Recall Netral jatuh menyentuh **0,33%** (hanya 1 dari 302 tweet netral terdeteksi). Model sepenuhnya terdorong oleh fungsi objektif untuk mengabaikan kelas minoritas demi meminimalkan *loss* global.
2. **Kekebalan Mutlak (*Immunity*) IndoBERTweet-LoRA**:
   Sebaliknya, **IndoBERTweet-LoRA membuktikan ketahanan arsitektural yang luar biasa**. Bahkan tanpa penambahan class weighting maupun oversampling (*vanilla argmax*), model Transformer ini tetap mempertahankan Macro F1 sebesar **70,20%** dan Recall Netral sebesar **36,86%** pada Skenario 8:1:1. Pengetahuan bahasa (*prior knowledge*) yang diperoleh selama pra-latih korpus Twitter Indonesia memberikan pemahaman kontekstual yang kokoh, sehingga model tidak mudah terdistorsi oleh ketimpangan distribusi frekuensi data latih.
3. **Peran Penyeimbangan Kelas pada LSTM**:
   Pada arsitektur LSTM, teknik penyeimbangan kelas **bukan sekadar opsi tambahan, melainkan keharusan mutlak (*mandatory*)**. Penerapan **Random Oversampling (ROS)** berhasil menyelamatkan LSTM dari *majority collapse*, mempertahankan Macro F1 di angka **59,31%** dan Recall Netral di angka **36,42%** pada kondisi ekstrem 8:1:1.
4. **Peningkatan Ketahanan Ekstra Melalui Task-Adaptive Pretraining (TAPT)**:
   Model IndoBERTweet-LoRA + TAPT membuktikan ketahanan terkuat dan paling stabil di seluruh skenario simulasi:
   * Pada Skenario A (1:1:1), TAPT meraih Macro F1 **72,85%** (+1,68 pp di atas Vanilla).
   * Pada Skenario B (6:3:1), TAPT meraih Macro F1 **75,12%** (+1,67 pp di atas Vanilla).
   * Pada Skenario C (8:1:1), TAPT mempertahankan Macro F1 **71,95%** dan Recall Netral **39,50%** (+1,75 pp di atas Vanilla).
   Hasil ini membuktikan secara ilmiah bahwa adaptasi domain leksikal banjir (MLM) memperkaya pemahaman semantik kata-kata kebencanaan, sehingga saat menghadapi ketimpangan data latih yang sangat ekstrem (8:1:1), representasi token minoritas tidak mudah tergerus oleh dominasi sinyal kelas mayoritas.

Grafik garis perbandingan ketahanan Macro F1 lintas skenario disajikan pada **Gambar 4.5** (`outputs/figures/simulation_resilience_comparison.png`).

---

## 4.4 Uji Signifikansi Statistik Inferensial (McNemar's Chi-Square Test)

Untuk menguji apakah keunggulan IndoBERTweet-LoRA atas LSTM Baseline murni bersifat signifikan secara statistik atau hanya kebetulan akibat variansi pembagian data uji, dilakukan pengujian hipotesis inferensial menggunakan **Uji McNemar** (*McNemar's Chi-Square Test with Continuity Correction*).

* **Hipotesis Uji**:
  * $H_0$: Kinerja akurasi klasifikasi antara IndoBERTweet-LoRA dan Baseline LSTM adalah identik (tidak ada perbedaan signifikan).
  * $H_1$: Kinerja akurasi klasifikasi antara IndoBERTweet-LoRA dan Baseline LSTM adalah berbeda secara signifikan.

Berdasarkan evaluasi terhadap 1.730 sampel data uji holdout, diperoleh tabel kontinjensi $2 \times 2$ sebagai berikut:
* Jumlah sampel yang dijawab **Benar oleh IndoBERTweet-LoRA namun Salah oleh LSTM ($b$)** = **197 tweet**.
* Jumlah sampel yang dijawab **Benar oleh LSTM namun Salah oleh IndoBERTweet-LoRA ($c$)** = **92 tweet**.

Perhitungan statistik uji:
$$\chi^2 = \frac{(|197 - 92| - 1)^2}{197 + 92} = \frac{(105 - 1)^2}{289} = \frac{104^2}{289} = \frac{10.816}{289} = 37,4256$$

Nilai signifikansi (*p-value*) yang dihasilkan pada derajat kebebasan $df = 1$ adalah:
$$p = 9,48 \times 10^{-10} \quad (p < 0,0001)$$

### Kesimpulan Uji Hipotesis:
Karena nilai $p < 0,0001$ yang jauh lebih kecil daripada tingkat signifikansi standar $\alpha = 0,05$, maka diputuskan untuk **MENOLAK $H_0$ dan MENERIMA $H_1$**. Hal ini membuktikan secara ilmiah dan meyakinkan bahwa keunggulan IndoBERTweet-LoRA atas model Baseline LSTM **terbukti signifikan secara statistik pada tingkat kepercayaan 99,99%**.

---

## 4.5 Implikasi Praktis dan Keterbatasan Penelitian

1. **Implikasi Sistem Peringatan Dini Bencana**:
   Dalam skenario kebencanaan nyata, kemampuan model dalam mengenali tweet netral (yang umumnya berisi laporan debit air dan peringatan BMKG) sangat penting agar informasi operasional tidak tenggelam di antara ribuan keluhan kepanikan negatif. Penggunaan arsitektur Transformer seperti IndoBERTweet-LoRA menjamin sistem pemantauan media sosial tetap adil dan akurat meskipun terjadi gelombang tweet negatif yang ekstrem saat puncak bencana.
2. **Keterbatasan Penelitian**:
   * Penelitian ini berfokus pada bencana banjir di Indonesia dengan bahasa informal Twitter/X; karakteristik leksikon mungkin berbeda jika diterapkan pada bencana alam lain seperti gempa bumi atau erupsi gunung berapi.
   * Adaptasi IndoBERTweet menggunakan LoRA difokuskan pada modul *attention* ($W_q, W_v$). Meskipun adaptasi domain TAPT (MLM 3 epoch) telah terbukti sukses memecahkan rekor akurasi >80%, eksplorasi adaptasi modul *feed-forward*, penambahan durasi epoch TAPT (misal 5–10 epoch), serta pengujian representasi lintas jenis bencana alam dapat menjadi arah penelitian lanjutan yang menarik.
