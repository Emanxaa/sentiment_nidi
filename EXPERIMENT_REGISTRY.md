# Master Experiment Registry — Thesis Sentiment Benchmark

**Project:** Indonesian Disaster Sentiment Analysis (LSTM vs IndoBERTweet-LoRA)  
**Dataset:** `Data/processed/banjir_processed_v2.csv` ($n=8,648$, 80:20 Stratified Partition)  
**Hardware Baseline:** Nvidia Tesla T4 (Kaggle Cloud)  
**Primary Evaluation Metric:** Macro F1-Score  

---

## 1. Official Master Experiment Log

| Experiment ID | Architecture | Text Input | Balancing Technique | LoRA Config | Learning Rate | Test Macro F1 (%) | Test Accuracy (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **LSTM-M3** | BiLSTM | `processed_text_v2` | None (Natural) | N/A | 1e-3 | 64.95% | 72.45% | **COMPLETED (LSTM Ref)** |
| **LSTM-M4** | BiLSTM | `processed_text_v2` | Class Weight | N/A | 1e-3 | 63.26% | 71.21% | **COMPLETED (LSTM Ref)** |
| **LSTM-M5** | BiLSTM | `processed_text_v2` | Random Oversampling | N/A | 1e-3 | 64.81% | 72.83% | **COMPLETED (LSTM Ref)** |
| **LSTM-M6** | BiLSTM | `processed_text_v2` | Random Undersampling| N/A | 1e-3 | 62.01% | 68.96% | **COMPLETED (LSTM Ref)** |
| **LSTM-M7** | BiLSTM | `processed_text_v2` | SMOTE (Sequence) | N/A | 1e-3 | 64.12% | 71.85% | **COMPLETED (LSTM Ref)** |
| **IB-B03-LEG**| IndoBERT-LoRA | `text_with_emoticon`| None (Natural) | r=16, a=32, d=0.3 | 2e-4 | 73.45% | 78.73% | **COMPLETED (Legacy Trial)** |
| **IB-B1-BASE**| IndoBERT-LoRA | `processed_text_v2` | None (Natural) | r=16, a=32, d=0.1 | 2e-5 | 51.72% | 69.04% | **COMPLETED (B1 Default, 3 Seeds)** |
| **B1.1-clean** | IndoBERT-LoRA | `clean_text` | None (Natural) | r=16, a=32, d=0.1 | 2e-5 | 29.66% (Val: 27.81%) | 56.47% | **COMPLETED (B1.1 Ablation)** |
| **B1.1-v2**    | IndoBERT-LoRA | `processed_text_v2` | None (Natural) | r=16, a=32, d=0.1 | 2e-5 | 31.34% (Val: 30.28%) | 57.05% | **COMPLETED (B1.1 Official Selected Input)** |
| **IB-B0.5-S2**| IndoBERT-LoRA | `processed_text_v2` | None (Natural) | r=16, a=32, d=0.1 | 1e-5 | — | — | PLANNED (B0.5 LR Sweep) |
| **IB-B0.5-S3**| IndoBERT-LoRA | `processed_text_v2` | None (Natural) | r=16, a=32, d=0.1 | 3e-5 | — | — | PLANNED (B0.5 LR Sweep) |
| **IB-B0.5-S4**| IndoBERT-LoRA | `processed_text_v2` | None (Natural) | r=8, a=16, d=0.05 | Tuning | — | — | PLANNED (B0.5 LoRA Sweep) |
| **IB-B0.5-S5**| IndoBERT-LoRA | `processed_text_v2` | None (Natural) | r=8, a=16, d=0.10 | Tuning | — | — | PLANNED (B0.5 LoRA Sweep) |
| **IB-B0.5-S6**| IndoBERT-LoRA | `processed_text_v2` | None (Natural) | r=16, a=32, d=0.05| Tuning | — | — | PLANNED (B0.5 LoRA Sweep) |
| **IB-B0.5-S7**| IndoBERT-LoRA | `processed_text_v2` | None (Natural) | r=32, a=64, d=0.05| Tuning | — | — | PLANNED (B0.5 LoRA Sweep) |
| **IB-B0.5-S8**| IndoBERT-LoRA | `processed_text_v2` | None (Natural) | r=32, a=64, d=0.10| Tuning | — | — | PLANNED (B0.5 LoRA Sweep) |
| **IB-B2-OPT** | IndoBERT-LoRA | Winning Input | None (Natural) | Optimal Config | Optimal | — | — | PLANNED (B2 Best Baseline) |
| **IB-B3-CW**  | IndoBERT-LoRA | Winning Input | Class Weight | Optimal Config | Optimal | — | — | PLANNED (B3 Balancing) |
| **IB-B4-ROS** | IndoBERT-LoRA | Winning Input | Random Oversampling | Optimal Config | Optimal | — | — | PLANNED (B4 Balancing) |
| **IB-B5-RUS** | IndoBERT-LoRA | Winning Input | Random Undersampling| Optimal Config | Optimal | — | — | PLANNED (B5 Balancing) |
| **IB-B6-SMOTE**| IndoBERT-LoRA | Winning Input | SMOTE (Hidden) | Optimal Config | Optimal | — | — | PLANNED (B6 Balancing) |
| **IB-B7-SYN** | Cross-Model | Master Dataset | All Strategies | All Architectures | — | — | — | PLANNED (B7 Synthesis & McNemar)|

---

## 2. Registry Update Protocol

1. **Immutability of Reference Rows:** Baseline LSTM rows (`LSTM-M3` through `LSTM-M7`) and historical benchmarks are locked and must never be altered.
2. **Automatic Metric Logging:** Future milestone execution scripts must populate the respective row upon passing verification checks.
3. **Status Taxonomy:**
   * `PLANNED`: Configuration specified; awaiting execution.
   * `RUNNING`: Kernel currently active on Kaggle GPU.
   * `COMPLETED`: 3 seeds verified, artifacts stored, report generated.
   * `REJECTED`: Gated failure (e.g. divergence or metric regression).
