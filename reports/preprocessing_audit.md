# Laporan Preprocessing Audit

**Sumber:** `Data/data_preprocessed_with_emoticon.csv` · **Jumlah baris:** 8648

## Audit Stopword (kata sinyal sentimen)

| Kata | Count input | Count output | Count stem | Status |
|---|---|---|---|---|
| tidak | 495 | 1580 | 1669 | OK |
| bukan | 407 | 432 | 673 | OK |
| jangan | 132 | 136 | 155 | OK |
| sedih | 174 | 189 | 301 | OK |
| marah | 16 | 20 | 129 | OK |
| doa | 151 | 225 | 360 | OK |
| harapan | 120 | 0 | 245 | TER-STEMMED |

## Audit LSTM

- Kasus negasi hilang: **0** (detail: `reports/lstm_negation_audit.csv`)
- Potensi stemming agresif: **8** (detail: `reports/lstm_stemming_audit.csv`)

## Audit IndoBERT

- Temuan: **5** (detail: `reports/bert_audit.csv`)

Catatan: `text_bert` (preprocess_bert) tidak menghapus noise seperti 'tampilkan lebih banyak', 'rb', dan timestamp — ini dikoreksi pada `data_banjir_v2.csv` lewat cleaning rules Task 0.4.
