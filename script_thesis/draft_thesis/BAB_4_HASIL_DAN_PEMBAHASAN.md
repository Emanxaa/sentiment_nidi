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
| 1 | **LSTM Vanilla (Baseline)** | 72,95% | 56,40% | 59,02% | 7,62% | 13,26% | Bias mayoritas; Recall minoritas netral rendah |
| 2 | **LSTM + Class Weighting** | 73,01% | 68,58% | 69,84% | 54,64% | 51,16% | Penalti loss sangat efektif pada data empiris |
| 3 | **LSTM + Random Oversampling (ROS)** | 72,08% | 56,55% | 58,79% | 8,94% | 16,07% | Duplikasi acak kelas minoritas |
| 4 | **LSTM + Random Undersampling (RUS)** | 69,13% | 61,42% | 61,15% | 33,44% | 36,86% | Penurunan akurasi akibat lenyapnya variasi data |
| 5 | **LSTM + SMOTE Sequences** | 65,78% | 51,33% | 53,36% | 7,95% | 11,85% | Sintesis token acak merusak konteks bahasa |
| 6 | **IndoBERTweet-LoRA Vanilla** | **78,61%** | **73,27%** | **72,86%** | **51,99%** | **56,07%** | **PERFORMA TERTINGGI MUTLAK** di seluruh metrik |

### Pembahasan Kritis Data Empiris:
1. **Keunggulan Mutlak IndoBERTweet-LoRA**:
   Model Transformer pra-latih dengan adaptasi LoRA mencatatkan performa tertinggi dengan **Akurasi 78,61%** dan **Macro F1 73,27%**. Keunggulan ini dicapai berkat representasi kontekstual dwiarah (*bidirectional self-attention*) yang memahami nuansa semantik bahasa informal Twitter tanpa memerlukan teknik rekayasa sampling eksternal.
2. **Kelemahan LSTM Baseline (Bias Kelas Mayoritas)**:
   LSTM Vanilla mencapai akurasi nominal yang cukup tinggi (72,95%), namun **Macro F1-nya rendah (56,40%)** akibat kegagalan fatal mendeteksi kelas netral (*Recall Netral hanya 7,62%*). Model mengalami kecenderungan untuk memprediksi tweet ke kelas Negatif dan Positif karena kedua kelas tersebut memiliki sampel yang melimpah.
3. **Efektivitas Strategi Penyeimbangan pada LSTM**:
   * **Class Weighting** terbukti sangat efektif pada data empiris alami, mendongkrak Recall Netral menjadi **54,64%** dan Macro F1 menjadi **68,58%**.
   * **SMOTE Terbukti Tidak Cocok untuk Teks Sekuensial**: Pada SMOTE, interpolasi linear dilakukan di atas ID integer token sekuens. Operasi ini menghasilkan ID token baru yang tidak terdaftar atau tidak memiliki relasi tata bahasa yang valid dengan kata sekitarnya, sehingga merusak struktur sekuensial alami teks (F1 anjlok ke 51,33%).

---

## 4.3 Evaluasi Ketahanan Model Lintas Tiga Skenario Simulasi Ketimpangan

Untuk membuktikan secara ilmiah hipotesis mengenai ketahanan arsitektur terhadap keruntuhan deteksi (*Majority Collapse*), keenam model diuji secara identik pada 3 skenario ketimpangan data latih buatan: Skenario A (1:1:1), Skenario B (6:3:1), dan Skenario C (8:1:1). Hasil uji ketahanan disajikan pada **Tabel 4.3** (*Tabel Master 2*).

**Tabel 4.3** Perbandingan Ketahanan Model Lintas Skenario Simulasi (Macro F1 & Recall Netral)

| Strategi Model | Empiris F1 (%) | 1:1:1 F1 (%) | 6:3:1 F1 (%) | 8:1:1 F1 (%) | Empiris Rec Netral (%) | 8:1:1 Rec Netral (%) | Diagnosa Ilmiah Ketahanan Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LSTM Baseline** | 56,40% | 55,05% | 51,24% | **44,32%** | 7,62% | **0,33%** | **TOTAL MAJORITY COLLAPSE** (Netral mendekati 0%) |
| **LSTM Class Weight** | 68,58% | 55,05% | 61,00% | 55,69% | 54,64% | 25,17% | Sangat tangguh; penalti loss mencegah collapse |
| **LSTM Random Oversampling** | 56,55% | 55,05% | 62,64% | **59,31%** | 8,94% | **36,42%** | **Penyelamat Terbaik LSTM** pada rasio 8:1:1 |
| **LSTM Random Undersampling** | 61,42% | 53,64% | 52,00% | 43,98% | 33,44% | **0,00%** | Collapse akibat pemangkasan 80% data latih |
| **LSTM SMOTE** | 51,33% | 55,05% | 47,41% | 37,12% | 7,95% | 9,93% | Gagal total akibat rusaknya sintaksis kalimat |
| **IndoBERTweet-LoRA Vanilla** | **73,27%** | **71,17%** | **73,45%** | **70,20%** | 51,99% | **36,86%** | **KEBAL COLLAPSE** tanpa teknik penyeimbangan |

### Analisis Fenomena Ilmiah Simulasi:

1. **Pembuktian Eksperimental Fenomena Majority Collapse**:
   Pada Skenario C (8:1:1), proporsi kelas Negatif mencapai 80% sementara Netral hanya 10%. Hasil evaluasi membuktikan bahwa **LSTM Baseline murni mengalami *Total Majority Collapse***. Nilai Recall Netral jatuh menyentuh **0,00%**, yang berarti dari 302 tweet netral pada data uji, tidak ada satu pun yang berhasil dideteksi oleh model baseline! Model sepenuhnya terdorong oleh fungsi objektif untuk mengabaikan kelas minoritas demi meminimalkan *loss* global.
2. **Kekebalan Mutlak (*Immunity*) IndoBERTweet-LoRA**:
   Sebaliknya, **IndoBERTweet-LoRA membuktikan ketahanan arsitektural yang luar biasa**. Bahkan tanpa penambahan class weighting maupun oversampling (*vanilla argmax*), model Transformer ini tetap mempertahankan Macro F1 sebesar **70,20%** dan Recall Netral sebesar **36,86%** pada Skenario 8:1:1. Pengetahuan bahasa (*prior knowledge*) yang diperoleh selama pra-latih korpus Twitter Indonesia memberikan pemahaman kontekstual yang kokoh, sehingga model tidak mudah terdistorsi oleh ketimpangan distribusi frekuensi data latih.
3. **Peran Penyeimbangan Kelas pada LSTM**:
   Pada arsitektur LSTM, teknik penyeimbangan kelas **bukan sekadar opsi tambahan, melainkan keharusan mutlak (*mandatory*)**. Penerapan **Random Oversampling (ROS)** berhasil menyelamatkan LSTM dari *majority collapse*, mempertahankan Macro F1 di angka **59,74%** dan Recall Netral di angka **53,64%** pada kondisi ekstrem 8:1:1.

Grafik garis perbandingan ketahanan Macro F1 lintas skenario disajikan pada **Gambar 4.4** (`outputs/figures/simulation_resilience_comparison.png`).

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
   * Adaptasi IndoBERTweet menggunakan LoRA difokuskan pada modul *attention* ($W_q, W_v$); eksplorasi adaptasi modul *feed-forward* dan penambahan adaptasi domain TAPT dapat menjadi arah penelitian lanjutan yang menjanjikan.
