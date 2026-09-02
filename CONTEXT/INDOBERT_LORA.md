# IndoBERTweet-LoRA Context & Architectural Contract

**Document Version:** 2.0.0  
**Authority:** Single Source of Truth for IndoBERTweet-LoRA Experiments  
**Target Runtime:** Kaggle Tesla T4 GPU (CUDA, Mixed Precision fp16)  

---

## 1. Objective

To fine-tune the pretrained `indolem/indobertweet-base-uncased` language model using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) for Indonesian flood disaster sentiment classification. 

The primary research objective is to conduct a fair, rigorous, and empirical benchmark of transformer representations against the completed LSTM baseline suite across natural imbalanced data and four balancing interventions (Class Weighting, ROS, RUS, SMOTE).

---

## 2. Dataset Specification

* **Canonical Dataset:** `Data/processed/banjir_processed_v2.csv`
* **Row Count:** Exactly 8,648 rows.
* **Primary Text Column:** `processed_text_v2` (LLM-completed, regex-refined, colloquial slang normalized).
* **Target Label Column:** `label`
  * `0`: Negative (4,686 samples / 54.19%)
  * `1`: Neutral (1,510 samples / 17.46%)
  * `2`: Positive (2,452 samples / 28.35%)

---

## 3. Split & Leakage Policy

* **Split Protocol:** Stratified Train-Val-Test partition strictly identical to the LSTM suite.
  * **Test Set:** 20% of total dataset ($n=1,730$), held out permanently. Never touched during training, validation, or checkpoint selection.
  * **Train Set:** 72% of total dataset ($n=6,226$).
  * **Validation Set:** 8% of total dataset ($n=692$, derived as 10% of the 80% combined training split).
* **Zero Leakage Invariant:** Subword tokenization and vocabulary caching must occur strictly *after* data partitioning. Validation and Test partitions must never be balanced, resampled, or augmented.

---

## 4. Model & LoRA Principles

* **Base Model:** `indolem/indobertweet-base-uncased` (BERT architecture, 12 layers, 768 hidden dimension, 12 heads).
* **PEFT Framework:** Hugging Face PEFT LoRA.
* **LoRA Architecture:**
  * Task Type: `SEQ_CLS` (Sequence Classification, 3 classes).
  * Target Modules: Attention projection matrices (`query`, `value`).
  * Rank ($r$): 16 (default baseline; tunable in B2).
  * Alpha ($lpha$): 32 ($2 \times r$).
  * Dropout: 0.1.
  * Modules to Save: `classifier` (linear classification head).
* **Parameter Boundary:** Base transformer backbone weights must be **100% frozen**. Trainable parameter budget must remain below 1.0% of total weights (~592k trainable out of 111.15M total). Full fine-tuning is strictly prohibited.

---

## 5. Training & GPU Policy

* **Accelerator:** Kaggle Tesla T4 GPU (16GB VRAM).
* **Precision:** Mixed Precision (`fp16=True`) with native PyTorch AMP.
* **Environment Guard:** Single GPU isolation (`CUDA_VISIBLE_DEVICES=0`), tokenizer parallelism disabled (`TOKENIZERS_PARALLELISM=false`).
* **Optimization:** AdamW optimizer (`weight_decay=0.01`, `warmup_ratio=0.1`).
* **Batch Size:** 16 (per device).
* **Sequence Length:** 128 tokens (post-padding and truncation).
* **Early Stopping:** Patience 2 epochs, monitoring validation `macro_f1`.
* **Checkpointing:** Best checkpoint restored automatically prior to multi-split evaluation.

---

## 6. Evaluation & Reporting Protocol

* **Primary Metric:** **Macro F1-Score** (harmonic mean of precision and recall unweighted across classes).
* **Secondary Metrics:** Accuracy, Macro Precision, Macro Recall, per-class Support.
* **Reproducibility Mandate:** Every milestone requires three independent random seeds (`42`, `123`, `456`).
* **Required Output Artifacts per Seed:**
  1. `best_model/` (PEFT adapter weights, config, and tokenizer)
  2. `history.csv` (per-epoch loss and accuracy)
  3. `metrics.json` (comprehensive multi-split metrics and runtime metadata)
  4. `loss_curve.png` & `accuracy_curve.png` (300 DPI learning curves)
  5. `confusion_train.png`, `confusion_val.png`, `confusion_test.png` (300 DPI heatmaps)
  6. `classification_report.csv` (per-class precision, recall, F1, support)
* **Aggregated Deliverables:**
  * `summary.csv`, `summary.json`, and automatic delta comparison against LSTM baseline in markdown report.
