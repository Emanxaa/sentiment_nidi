# BAB III: METODOLOGI PENELITIAN

---

## 3.1 Desain Penelitian dan Alur Kerja Sistem

Penelitian ini menggunakan pendekatan eksperimental kuantitatif berbasis pembelajaran mendalam (*deep learning*) dan pengolahan bahasa alami (*Natural Language Processing* / NLP). Fokus utama penelitian adalah mengevaluasi kinerja dan ketahanan arsitektur *Recurrent Neural Network* (LSTM) dan model berbasis Transformer (*IndoBERTweet*) yang diadaptasi menggunakan *Low-Rank Adaptation* (LoRA) dalam melakukan klasifikasi sentimen 3-kelas (Negatif, Netral, Positif) pada tweet kebencanaan banjir di Indonesia.

Alur komprehensif penelitian terbagi dalam empat tahapan utama:
1. **Pengadaan dan Pembersihan Data (Preprocessing Modular)**: Pembersihan regex komprehensif, normalisasi leksikon kata gaul/slang, dan standardisasi label sentimen.
2. **Partisi Data Bebas Kebocoran (*Zero Data Leakage*)**: Pembagian partisi data train, validation, dan test set dengan kunci acak terkunci `seed=42`.
3. **Pembentukan 3 Skenario Simulasi Ketimpangan Data Latih**: Pembangunan skenario Seimbang (1:1:1), Moderat (6:3:1), dan Ekstrem (8:1:1) dari partisi data latih murni.
4. **Eksperimen Pemodelan & Uji Signifikansi**: Pelatihan varian LSTM (Vanilla, Class Weighting, ROS, RUS, SMOTE) dan IndoBERTweet-LoRA Vanilla, evaluasi matriks konfusi, dan uji inferensial McNemar's Test.

---

## 3.2 Dataset dan Karakteristik Data

Dataset yang digunakan dalam penelitian ini adalah dataset tweet bencana banjir di Indonesia versi revisi (`data_v2`), dengan total korpus sebanyak **8.648 tweet**. Data ini merekam opini, laporan situasi, keluhan, maupun simpati publik masyarakat Indonesia selama periode kejadian bencana banjir di berbagai wilayah (seperti Sumatera Utara, Sumatera Barat, Riau, Jabodetabek, dan Jawa Tengah).

Setiap sampel data terdiri atas atribut teks tweet (`text`) dan anotasi sentimen manusia (`sentimen` / `label`) yang dikategorikan ke dalam 3 kelas:
1. **Kelas 0 (Negatif)**: Keluhan masyarakat, kepanikan, kritik atas lambatnya bantuan, kerusakan infrastruktur, dan kerugian materiil.
2. **Kelas 1 (Netral)**: Informasi faktual tinggi muka air, laporan pengalihan arus lalu lintas, update cuaca BMKG, dan berita evakuasi tanpa muatan emosional.
3. **Kelas 2 (Positif)**: Ucapan syukur banjir surut, apresiasi tim evakuasi SAR/TNI/Polri, penyaluran logistik makanan, dan doa keselamatan.

---

## 3.3 Tahapan Pembersihan Data (*Preprocessing*)

Data teks yang bersumber dari media sosial Twitter/X memiliki tingkat derau (*noise*) yang sangat tinggi. Oleh karena itu, diterapkan pipeline pembersihan modular non-destruktif:

1. **Pembersihan Berbasis Regex (*Regex Cleaning*)**:
   * Menghilangkan tautan internet (*URL HTTP/HTTPS* dan *domain WWW*).
   * Menghilangkan tanda *User Mention* (`@username`).
   * Menghilangkan simbol pagar pada tagar (`#banjir` diubah menjadi kata biasa `banjir` agar makna semantiknya tidak hilang).
   * Menghilangkan artefak antarmuka scraping web (misal: frasa "*tampilkan lebih banyak*", "*membalas*", "*baca berita*", dan angka metrik interaksi trailing seperti "*2 rb*", "*35 rb*").
   * Normalisasi spasi putih berlebih dan pemangkasan (*trimming*).

2. **Standardisasi Kata Gaul/Informal (*Slang Normalization*)**:
   * Menggunakan kamus leksikon formal-informal bahasa Indonesia baku (*Colloquial Indonesian Lexicon*). Kata-kata singkatan gaul (seperti *bgt* -> *sangat*, *bapack* -> *bapak*, *gabisa* -> *tidak bisa*, *bapake* -> *bapak*) dipetakan ke kosakata baku untuk mereduksi ukuran variasi kosakata (*out-of-vocabulary reduction*).

3. **Standardisasi Format Label**:
   * Label sentimen dikonversi ke dalam format integer terstandarisasi: Negatif dienkode menjadi `0`, Netral menjadi `1`, dan Positif menjadi `2`.

Hasil pembersihan disimpan ke dalam file kanonik `script_thesis/data/data_v2.csv`.

---

## 3.4 Partisi Data Bebas Kebocoran (*Zero Data Leakage Split*)

Untuk menjamin validitas dan reproduksibilitas pengujian ilmiah, dataset dibagi menggunakan metode *Stratified Random Sampling* dengan parameter pengunci acak deterministik `random_state = 42`. 

Rasio pembagian data dirancang sebagai berikut:
* **Data Uji Holdout (*Test Set*)**: 20% dari total dataset ($n = 1.730$ tweet). Data uji ini **dikunci secara mutlak** dan tidak pernah tersentuh oleh proses fitting tokenizer, resampling data, maupun pelatihan bobot model apapun. Seluruh model dan simulasi diuji pada data uji yang persis sama.
* **Data Validasi (*Validation Set*)**: 8% dari total dataset ($n = 692$ tweet), digunakan untuk evaluasi konvergensi *early stopping*.
* **Data Latih Murni (*Train Set*)**: 72% dari total dataset ($n = 6.226$ tweet), digunakan sebagai basis pembelajaran model.

Tabel 3.1 merinci distribusi sampel pada masing-masing partisi data:

**Tabel 3.1** Distribusi Partisi Data Latih, Validasi, dan Uji

| Partisi Data | Negatif (0) | Netral (1) | Positif (2) | Total Tweet | Persentase |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | 3.376 | 1.087 | 1.763 | **6.226** | 72,0% |
| **Validation Set** | 375 | 121 | 196 | **692** | 8,0% |
| **Test Set (Holdout)** | 937 | 302 | 491 | **1.730** | 20,0% |
| **Total Korpus** | **4.688 (54,2%)** | **1.510 (17,5%)** | **2.450 (28,3%)** | **8.648** | **100,0%** |

---

## 3.5 Desain Tiga Skenario Simulasi Ketimpangan Data Latih

Masalah utama dalam analisis sentimen media sosial adalah ketimpangan kelas (*class imbalance*). Untuk menguji ketahanan model secara eksperimental dan memicu fenomena *Majority Collapse*, penelitian ini merancang **3 Skenario Simulasi Data Latih** yang disampel secara terkontrol dari data latih murni (`train.csv`):

1. **Skenario A: Seimbang Buatan (Rasio 1:1:1)**
   * Kuota sampel: 1.000 Negatif : 1.000 Netral : 1.000 Positif (Total = 3.000 sampel).
   * Bertujuan sebagai kontrol dasar (*ideal benchmark*) untuk melihat bagaimana performa model apabila data latih memiliki prior kelas yang identik sempurna.
2. **Skenario B: Ketimpangan Moderat (Rasio 6:3:1)**
   * Kuota sampel: 3.000 Negatif : 500 Netral : 1.500 Positif (Total = 5.000 sampel).
   * Bertujuan merefleksikan kondisi ketimpangan menengah di mana kelas Netral mengalami penyusutan menjadi minoritas.
3. **Skenario C: Ketimpangan Ekstrem / Ekor Panjang (Rasio 8:1:1)**
   * Kuota sampel: 3.200 Negatif : 400 Netral : 400 Positif (Total = 4.000 sampel).
   * Bertujuan menguji kondisi stres kritis (*stress-test*) apakah model mengalami *Majority Collapse* (kehilangan kemampuan mengenali kelas minoritas akibat dominasi 80% kelas mayoritas negatif).

Ketiga skenario disimpan masing-masing ke dalam file `scenario_111.csv`, `scenario_631.csv`, dan `scenario_811.csv`.

---

## 3.6 Arsitektur Model dan Strategi Penanganan Ketimpangan

### 3.6.1 Arsitektur Model LSTM
Model *Long Short-Term Memory* (LSTM) dibangun menggunakan spesifikasi terstandarisasi:
* **Embedding Layer**: Ukuran kosakata (*vocab size*) = 10.000 token, dimensi representasi vektor = 128.
* **LSTM Layer**: 64 unit sel memori LSTM untuk menangkap dependensi sekuensial teks.
* **Dropout Regularization**: Rate = 0.3 untuk mencegah overfitting.
* **Dense Output Layer**: 3 unit neuron dengan fungsi aktivasi *Softmax* untuk menghasilkan probabilitas distribusi 3 kelas.
* **Optimizer**: Adam dengan laju pembelajaran (*learning rate*) $lpha = 0,0005$, fungsi *loss* berupa *sparse categorical crossentropy*, ukuran batch = 32, dan *Early Stopping* dengan kesabaran (*patience*) = 3 epoch terhadap *validation loss*.

Untuk mengatasi ketimpangan kelas, diuji 5 varian perlakuan data latih pada LSTM:
1. **LSTM Vanilla (Natural Baseline)**: Pelatihan tanpa manipulasi data latih.
2. **LSTM Class Weighting**: Pembobotan kerugian inversi frekuensi:
   $$w_c = \frac{N}{K \cdot n_c}$$
   di mana $N$ adalah total sampel, $K$ adalah jumlah kelas (3), dan $n_c$ adalah frekuensi kelas $c$.
3. **LSTM Random Oversampling (ROS)**: Melakukan duplikasi acak pada kelas minoritas hingga menyamai kelas mayoritas.
4. **LSTM Random Undersampling (RUS)**: Memangkas sampel kelas mayoritas hingga seimbang dengan kelas minoritas.
5. **LSTM Synthetic Minority Over-sampling Technique (SMOTE)**: Melakukan interpolasi vektor kontinu pada ruang sekuens token integer untuk menciptakan sampel sintetik.

### 3.6.2 Arsitektur Model IndoBERTweet-LoRA
Model IndoBERTweet merupakan model bahasa pra-latih Transformer (*encoder-only*) yang dilatih khusus pada korpus tweet bahasa Indonesia oleh IndoLEM. Untuk melakukan adaptasi tugas hilir secara efisien tanpa melatih ulang seluruh 124 juta parameter dasar, diterapkan **Low-Rank Adaptation (LoRA)**:

* **Mekanisme Pembobotan LoRA**: Bobot matriks pra-latih $W_0 \in \mathbb{R}^{d \times k}$ dibekukan (*frozen*), dan modifikasi ditambahkan melalui dekomposisi matriks peringkat rendah:
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
  di mana $A \in \mathbb{R}^{r \times k}$ diinisialisasi secara Gaussian, $B \in \mathbb{R}^{d \times r}$ diinisialisasi nol, peringkat rank $r = 16$, skala $\alpha = 32$, dan modul target adaptasi difokuskan pada matriks perhatian *query* dan *value*.
* **Hyperparameter Pelatihan**: Laju pembelajaran (*learning rate*) $\alpha = 2 \times 10^{-4}$, ukuran batch = 16, dropout = 0.3, panjang sekuens maksimum = 128 token, dan evaluasi berbasis Macro F1.

---

## 3.7 Metrik Evaluasi dan Uji Signifikansi Statistik

Kinerja model dievaluasi secara menyeluruh menggunakan metrik standar klasifikasi teks:
1. **Accuracy**: Rasio prediksi benar terhadap total sampel.
2. **Macro F1-Score**: Rata-rata harmonik F1-score tanpa memandang bobot jumlah sampel kelas, sangat krusial untuk mengukur keadilan model terhadap kelas minoritas:
   $$\text{Macro F1} = \frac{1}{3} \sum_{c=0}^{2} F1_c$$
3. **Recall Minoritas (Netral)**: Kemampuan model dalam menemukan kembali seluruh sampel kelas netral yang sebenarnya:
   $$\text{Recall}_{\text{Netral}} = \frac{TP_{\text{Netral}}}{TP_{\text{Netral}} + FN_{\text{Netral}}}$$
4. **Uji Signifikansi Inferensial McNemar (*McNemar's Chi-Square Test*)**:
   Untuk menguji apakah perbedaan akurasi antara IndoBERTweet-LoRA dan Baseline LSTM signifikan secara statistik atau hanya kebetulan acak, dihitung tabel kontinjensi $2 \times 2$ dengan statistik uji bertransformasi koreksi kontinuitas Edwards:
   $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
   di mana $b$ adalah jumlah sampel yang dijawab benar oleh model A namun salah oleh model B, dan $c$ adalah sebaliknya. Derajat kebebasan ($df$) = 1, dengan tingkat signifikansi $\alpha = 0,05$.
