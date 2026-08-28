# Laporan Evaluasi Kualitas Label (Phase 5)

- Jumlah sampel: **1000**
- Agreement rate: **80.20%** (802/1000)
- Cohen's Kappa: **0.6734** — interpretasi: **Baik**

## Confusion Matrix (baris=label asli, kolom=label LLM)

| | negatif | netral | positif |
|---|---|---|---|
| negatif | 458 | 40 | 62 |
| netral | 18 | 127 | 24 |
| positif | 20 | 34 | 217 |

## Interpretasi Kappa

| Kappa | Interpretasi |
|---|---|
| <0.40 | Buruk |
| 0.40-0.60 | Sedang |
| 0.60-0.80 | Baik |
| >0.80 | Sangat baik |

## Label Flip Analysis

- Total label berubah: **198**
- Detail per baris: `Annotation/reports/label_flip_analysis.csv`

### Sebaran arah perubahan

- negatif -> positif: 62
- negatif -> netral: 40
- positif -> netral: 34
- netral -> positif: 24
- positif -> negatif: 20
- netral -> negatif: 18

### Sebaran kategori (heuristik, siap disempurnakan manual)

- Lainnya: 110
- Sarkasme: 42
- Informational: 25
- Mixed sentiment: 10
- Apresiasi: 8
- Kritik: 3
