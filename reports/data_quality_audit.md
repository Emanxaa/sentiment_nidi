# Laporan Data Quality Audit

**Sumber:** `Data/data_banjir.csv` · **Jumlah baris:** 8648

## Ringkasan vs Success Metrics

| Metrik | Hasil | Target |
|---|---|---|
| Missing value | 98.09% (tertinggi: emoticon) | <1% |
| Duplicate (exact+repost) | 0.00% (0 baris) | <5% |
| Near duplicate | 183 pasangan | — |
| Noise (baris kena pola) | 5842 kemunculan pola | <3% |

## Task 0.1 — Duplicate Audit

- Exact duplicate: **0** baris
- Repost (teks sama, tanggal beda): **0** baris
- Near duplicate: **183** baris
- Detail: `reports/duplicate_report.csv`

## Task 0.2 — Missing Value Audit

| Kolom | Missing | Persentase |
|---|---|---|
| text | 0 | 0.00% |
| sentimen | 0 | 0.00% |
| emoticon | 8483 | 98.09% |
| keyword | 0 | 0.00% |
| created_at | 0 | 0.00% |

## Task 0.3 — Noise Detection

| Pola | Jumlah baris | Persentase |
|---|---|---|
| tampilkan_lebih_banyak | 858 | 9.92% |
| engagement_rb | 8 | 0.09% |
| timestamp_video | 111 | 1.28% |
| sumber_berita | 877 | 10.14% |
| noise_token | 3970 | 45.91% |
| artefak_huruf_matematika | 18 | 0.21% |

## Task 0.4 — Cleaning Rules (contoh)

Contoh sebelum/sesudah penerapan aturan baru (lengkap: `reports/cleaning_rules_preview.csv`):

- **id 5720**
  - sebelum: `Pak Mualem mengerahkan lima orang relawan asal China untuk membantu pencarian korban banjir dan longsor yang masih tertimbun. Para relawan disebut memiliki alat untuk melacak jenazah .  Semoga Aceh segera pulih 4 11 24 531`
  - sesudah: `Pak Mualem mengerahkan lima orang relawan asal China untuk membantu pencarian korban banjir dan longsor yang masih tertimbun. Para relawan disebut memiliki alat untuk melacak jenazah . Semoga Aceh segera pulih 4 11 24 531`
- **id 4103**
  - sebelum: `Keluhkan Fasilitas yang Rusak Parah Imbas Banjir Bandang, Warga Aceh Tamiang Bandingkan Kondisi saat Tsunami di 2004 Silam matapersindonesia.com Keluhkan Fasilitas yang Rusak Parah Imbas Banjir Bandang, Warga Aceh Tamiang Bandingkan Kondisi... Menyor`
  - sesudah: `Keluhkan Fasilitas yang Rusak Parah Imbas Banjir Bandang, Warga Aceh Tamiang Bandingkan Kondisi saat Tsunami di 2004 Silam matapersindonesia. Keluhkan Fasilitas yang Rusak Parah Imbas Banjir Bandang, Warga Aceh Tamiang Bandingkan Kondisi... Menyoroti`
- **id 222**
  - sebelum: `Pemerintah Kabupaten Aceh Utara kembali menetapkan status tanggap darurat bencana alam menyusul terjadinya banjir susulan yang melanda sejumlah wilayah. Status tanggap darurat ini berlaku selama 15 hari ke depan mulai dari 10 hingga 24 Januari mendat`
  - sesudah: `Pemerintah Kabupaten Aceh Utara kembali menetapkan status tanggap darurat bencana alam menyusul terjadinya banjir susulan yang melanda sejumlah wilayah. Status tanggap darurat ini berlaku selama 15 hari ke depan mulai dari 10 hingga 24 Januari mendat`
- **id 4832**
  - sebelum: `Garoga banjir lagi. Semoga segera surut airnya ya ALLAH Dari  Aida Greenbury 4 35 61 1 rb`
  - sesudah: `Garoga banjir lagi. Semoga segera surut airnya ya ALLAH Dari Aida Greenbury 4 35 61`
- **id 5100**
  - sebelum: `Membalas  @DokterTifa Saya prediksi penyebab banjir bandang di sumatra itu, tiupan angin Tdk searah (arah angin yg berlawanan) hingga awan tidak bisa maju dan mundur. Pada khirnya hujan bertahan 1 tempat dan mengakibatkan  Banjir.Tip; Hentikan hujan:`
  - sesudah: `@DokterTifa Saya prediksi penyebab banjir bandang di sumatra itu, tiupan angin Tdk searah (arah angin yg berlawanan) hingga awan tidak bisa maju dan mundur. Pada khirnya hujan bertahan 1 tempat dan mengakibatkan Banjir.Tip; Hentikan hujan: azan terus`
