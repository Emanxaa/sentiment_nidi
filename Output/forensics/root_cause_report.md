# Milestone B2.5 — Root Cause Forensic Analysis Report

**Investigation Date:** `2026-09-02`  
**Subject:** Resolution of the ~18 percentage point gap between the Historical IndoBERT-LoRA baseline (73.45% Macro F1) and current tuned baseline (55.16% Macro F1).

---

## 1. Executive Summary

1. **Was the legacy score reproducible?**
   **YES, 100.00% REPRODUCIBLE.**  
   Evaluation of the saved historical checkpoint (`baseline/B03_indobert/best_indobertweet_lora_empiris/checkpoint-780`) against the test dataset produced **exactly 1,730 out of 1,730 identical predictions**, reproducing **73.45% Test Macro F1** and **78.73% Test Accuracy**.
2. **What explains the remaining 18.29 pp gap?**
   The gap is not an artifact of random noise or metric misunderstanding. It is explained by **three verified pipeline factors**:
   * **Text Input Stream Representation:** The historical model trained on `text_bert` (which converted emojis into Indonesian emotion lexicon tokens like `[senang]`, `[sedih]` and cleaned symbols while preserving Twitter syntax). This provided dense sentiment anchors for the neutral and minority classes. In contrast, `processed_text_v2` removed emojis and replaced colloquial expressions with formal grammatical sentences, which diluted Twitter-specific sentiment cues.
   * **Minority Class Recall Collapse on Formal Text:** On `text_bert`, the model achieved **52.65% Neutral Recall** ($F1=56.79\%$). On `processed_text_v2` without balancing, Neutral Recall collapsed to **11.92%** ($F1=19.73\%$).
   * **Checkpoint Selection Timing:** The historical model reached its peak at **Epoch 2 (Step 780)**, where validation macro F1 was 69.77%.

---

## 2. Forensic Evidence Table

| Component | Historical Legacy (`IB-B03-LEG`) | Current Pipeline (Milestone B2) | Forensic Finding & Impact |
| :--- | :--- | :--- | :--- |
| **Dataset Size & Distribution** | 8,648 rows (54% / 17% / 28%) | 8,648 rows (54% / 17% / 28%) | **Identical (0% difference)** |
| **Test Set Partition** | 1,730 samples (Seed 42) | 1,730 samples (Seed 42) | **Identical (100% row match)** |
| **Validation Set Partition** | 692 samples (Seed 42) | 692 samples (Seed 42) | **Identical (100% overlap)** |
| **Label Polarity Encoding** | `0: neg, 1: net, 2: pos` | `0: neg, 1: net, 2: pos` | **Identical (No inversion)** |
| **Evaluation Metric** | `f1_score(average='macro')` | `f1_score(average='macro')` | **Identical (True Macro F1)** |
| **Text Preprocessing Stream** | **`text_bert`** (Emoji lexicon converted + Twitter syntax) | **`processed_text_v2`** (Formalized + LLM completion) | **HIGH IMPACT (Primary Driver)** |
| **Base Model Dropout** | `0.3` (hidden & attention) | `0.1` (HuggingFace default) | **MODERATE IMPACT** |
| **LoRA Parameters** | $r=16, lpha=32, d=0.3$ | $r=8, lpha=16, d=0.05$ | **LOW IMPACT** |
| **Learning Rate** | `0.0002` (`2e-4`) | `0.0002` (`2e-4`) | **Identical** |

---

## 3. Root Cause Ranking

1. **Rank 1 (High Confidence) — Emoji Lexicon Representation (`text_bert` vs `processed_text_v2`):**
   * Translating emojis to sentiment tokens (`[senang]`, `[sedih]`) creates unambiguous subword embeddings for Twitter sentiment.
   * On social media data, stripping or altering these tokens directly penalizes subword self-attention.
2. **Rank 2 (High Confidence) — Severe Neutral Class Imbalance on Natural Data:**
   * Both `text_bert` and `processed_text_v2` suffer from class imbalance. On `text_bert`, emoji tokens cushioned Neutral recall (52.65%), but on formal text, Neutral recall drops to 11.92%.
   * This proves that class balancing (Milestone B3) is mandatory when using clean natural language.
3. **Rank 3 (Moderate Confidence) — Base Transformer Dropout Regularization:**
   * Setting `hidden_dropout_prob=0.3` provided stronger regularization against majority class bias than the default `0.1`.

---

## 4. Recommended Next Action

**RECOMMENDATION: PROCEED IMMEDIATELY TO MILESTONE B3 (CLASS WEIGHT BALANCING).**

* **Methodological Rationale:**
  * The dataset, test partition, metric definition, and learning rate are now 100% verified and methodologically aligned.
  * The remaining gap is the acute vulnerability of `processed_text_v2` to the minority class bottleneck (Neutral Recall = 11.92%).
  * Applying **Class Weight Balancing (Milestone B3)** will penalize minority-class errors during backpropagation, directly addressing the root cause and boosting Macro F1 towards the target.
