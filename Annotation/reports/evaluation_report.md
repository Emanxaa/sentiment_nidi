# Laporan Evaluasi Kualitas Label (Phase 5)

- Jumlah sampel: **1000**
- Agreement rate: **65.80%** (658/1000)
- Cohen's Kappa: **0.4609** — interpretasi: **Sedang**

## Confusion Matrix (baris=label asli, kolom=label LLM)

| | negatif | netral | positif |
|---|---|---|---|
| negatif | 349 | 91 | 120 |
| netral | 33 | 91 | 45 |
| positif | 21 | 32 | 218 |

## Interpretasi Kappa

| Kappa | Interpretasi |
|---|---|
| <0.40 | Buruk |
| 0.40-0.60 | Sedang |
| 0.60-0.80 | Baik |
| >0.80 | Sangat baik |

## Label Flip Analysis

- Total label berubah: **342**
- Detail per baris: `Annotation/reports/label_flip_analysis.csv`

### Sebaran arah perubahan

- negatif -> positif: 120
- negatif -> netral: 91
- netral -> positif: 45
- netral -> negatif: 33
- positif -> netral: 32
- positif -> negatif: 21

### Sebaran kategori (heuristik, siap disempurnakan manual)

- Lainnya: 191
- Sarkasme: 53
- Informational: 51
- Mixed sentiment: 19
- Apresiasi: 17
- Kritik: 11
