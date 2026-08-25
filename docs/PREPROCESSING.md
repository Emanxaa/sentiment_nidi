# Pipeline Preprocessing

## Alur Preprocessing

```
Data Mentah (data_banjir.csv)
         |
         v
    +----------+
    |   EDA    |  ----> Distribusi sentimen, WordCloud
    +----------+
         |
         v
    +------------------+
    | Emoticon Handling | ----> Gabungkan teks + emoticon, konversi emoticon ke kata
    +------------------+
         |
         v
    +------------------------+
    | Preprocessing LSTM     | ----> Lowercase, URL/mention removal, hashtag, normalisasi, stopword, stemming
    +------------------------+
         |
    +------------------------+
    | Preprocessing IndoBERT | ----> Lowercase, URL/mention removal, hashtag, rapikan simbol
    +------------------------+
         |
         v
    +------------------+
    | Label Encoding   | ----> negatif=0, netral=1, positif=2
    +------------------+
         |
         v
    +------------------+
    | Train-Test Split | ----> 80% train, 20% test (stratified)
    +------------------+
         |
         v
    Data Siap (split_data.pkl)
```

## Detail Preprocessing LSTM

1. **Lowercase**: Semua teks diubah ke lowercase
2. **URL & Mention Removal**: Menghapus URL (`http...`, `www...`) dan mention (`@user`)
3. **Hashtag**: Hashtag dipertahankan sebagai kata (`#banjir` -> `banjir`)
4. **Noise Removal**: Menghapus kata scraping noise (`com`, `rb`, `tampilkan`, `membalas`)
5. **Karakter Non-huruf**: Menghapus semua karakter selain huruf dan spasi
6. **Tokenizing**: Memecah teks menjadi token
7. **Normalisasi**: Mengganti slang (contoh: `gk` -> `tidak`, `gw` -> `saya`)
8. **Stopword Removal**: Menghapus stopwords sambil mempertahankan kata sinyal sentimen
9. **Stemming**: Menggunakan Sastrawi stemmer (dengan cache)

## Detail Preprocessing IndoBERT

1. **URL & Mention Removal**: Sama seperti LSTM
2. **Hashtag**: Dipertahankan sebagai kata
3. **Simbol Rapikan**: Menghapus/tambahkan spasi pada simbol yang mengganggu
4. **Tidak ada stemming/normalisasi**: Karena model transformer sensitif terhadap konteks asli

## Stopwords Khusus

### keep_words (LSTM)
Kata-kata yang tidak dihapus karena punya sinyal sentimen:
`tidak`, `bukan`, `jangan`, `belum`, `sedih`, `marah`, `senang`, `doa`, `dukungan`, `setuju`, `semangat`, `tertawa`, `bingung`, `haru`, `pulih`, `peringatan`, `darurat`, `hujan`, `badai`, `lokasi`, `informasi`, `harapan`, `takut`

### custom_stopwords (WordCloud)
Stopwords tambahan untuk visualisasi WordCloud termasuk lokasi Sumatera (`sumut`, `sumbar`, `sumsel`, `riau`, `jambi`, `lampung`) dan kata umum (`banjir`, `warga`, `korban`, `air`, `hujan`, `wilayah`)

## Normalisasi

Dictionary normalisasi mengubah slang bahasa Indonesia ke kata baku:
- `gk`, `ga`, `nggak`, `ngga`, `tdk`, `tak` -> `tidak`
- `gw`, `gue`, `w` -> `saya`
- `lo`, `lu` -> `kamu`
- `bgt`, `bangett` -> `banget`
- `dr`, `dri` -> `dari`
- `yg` -> `yang`
- `dgn` -> `dengan`
- `krn`, `karna` -> `karena`
- `tp`, `tpi` -> `tapi`
- `jd` -> `jadi`
- `udh`, `udah` -> `sudah`
- `sm` -> `sama`
- `aja` -> `saja`
- `utk` -> `untuk`
- `dlm` -> `dalam`
- `org` -> `orang`

## Emoticon Handling

Emoticon dikonversi menjadi kata-kata dalam bahasa Indonesia:
- `😭`, `😢`, `🥲`, `🥺`, `😔`, `😞` -> `sedih`
- `💔` -> `sedih`, `❤️‍🩹` -> `pulih`
- `🙏`, `🙏🏻`, `🤲` -> `doa`
- `💪` -> `semangat`, `👍`, `✅` -> `setuju`
- `🤝` -> `kerja sama`, `💖`, `❤️`, `❤`, `🤍`, `💚` -> `dukungan`
- `😊`, `☺️`, `🥰` -> `senang`
- `🚨` -> `darurat`, `‼️` -> `peringatan`
- `📢` -> `informasi`, `📍` -> `lokasi`
- `🌧️` -> `hujan`, `🌀` -> `badai`
- `🤔`, `🙃` -> `bingung`
