# Ringkasan Eksekutif Hasil Komparasi Lengkap (Data V2)
- **Rekor Tertinggi Keseluruhan**: IndoBERTweet-LoRA + TAPT (Akurasi 80.06%, Macro F1 74.61%).
- **P1 Sweep Optimal**: IndoBERTweet-LoRA Sweep t10 (Akurasi 77.98%, Macro F1 73.9%).
- **Ketahanan pada 8:1:1**: IndoBERTweet-LoRA mempertahankan Macro F1 di atas 70% (Bebas Collapse).
- **Penyelamat LSTM pada 8:1:1**: Random Oversampling (ROS) mempertahankan F1 59.31% dan Recall Netral 36.42%.
- **Signifikansi Statistik**: McNemar chi2 = 37.43, p < 0.0001 (Signifikan pada taraf 99.99%).
