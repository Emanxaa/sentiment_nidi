# IndoBERTweet-LoRA Empirical Failure Diagnosis & Root Cause Analysis

**Document Version:** 1.0.0  
**Date:** 2026-09-02  
**Role:** ML Research Engineer  
**Status:** Comprehensive Post-Mortem & Diagnostic Blueprint  

---

## 1. Executive Summary

Empirical benchmarking of the official **IndoBERTweet-LoRA** baseline under natural class imbalance revealed a pronounced performance discrepancy:
* **LSTM Reference Baseline (Milestone M3):** Accuracy **72.45%**, Macro F1 **64.95%**.
* **IndoBERTweet-LoRA (Default Parameter Config, $lr=2\times 10^{-5}$):** Accuracy **69.04%**, Macro F1 **51.72%** ($\Delta = -13.23\text{ pp}$).
* **Legacy Trial 4 (`B03_empiris`, $lr=2\times 10^{-4}$):** Accuracy **78.73%**, Macro F1 **73.45%** ($+8.50\text{ pp}$ over LSTM).

This diagnostic document investigates the structural and parametric root causes behind this 21.73 pp divergence between the underperforming configuration and the high-performing configuration. The analysis strictly separates **observed evidence**, **falsifiable hypotheses**, and **recommended validation experiments**.

---

## 2. Observed Empirical Evidence

The repository contains concrete, verifiable measurements recorded in `baseline/B03_indobert/trainer_state.json`, `docs/P0_VERIFIKASI_EVALUASI.md`, and `Output/indobert_lora/summary.json`:

### Evidence A: Hyperparameter Tuning History (docs/P0_VERIFIKASI_EVALUASI.md)
During previous 6-trial grid tuning on IndoBERTweet-LoRA, learning rate was the single dominant factor controlling Macro F1:

| Trial | Learning Rate | LoRA $r$ | LoRA $\alpha$ | Dropout | Val Accuracy | Val Macro F1 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Trial 4** | **$2\times 10^{-4}$** | **16** | **32** | **0.3** | **72.11%** | **0.7233** | **Optimal Configuration** |
| Trial 2 | $1\times 10^{-4}$ | 8 | 16 | 0.2 | 71.10% | 0.5878 | Sub-optimal |
| Trial 5 | $1\times 10^{-4}$ | 16 | 32 | 0.2 | 70.95% | 0.5341 | Minority Suppression |
| Trial 3 | $1\times 10^{-4}$ | 16 | 32 | 0.3 | 69.22% | 0.5036 | Minority Suppression |
| Trial 1 | **$5\times 10^{-5}$** | 8 | 16 | 0.2 | 64.16% | **0.4559** | **Severe Underfitting** |
| Trial 6 | $1\times 10^{-4}$ | 16 | 32 | 0.3 | 63.29% | 0.4531 | Batch Size 32 underflow |

### Evidence B: Parameter Checkpoint Verification
* Inspection of `baseline/B03_indobert/best_indobertweet_lora_empiris/checkpoint-780/trainer_state.json` (lines 14, 63, 131) verifies that the 73.45% Macro F1 model trained with `learning_rate = 0.0001948` (initial $2\times 10^{-4}$ linear decay).
* In contrast, the baseline execution configured with $lr = 2\times 10^{-5}$ (0.00002) produced an average Macro F1 of **51.72%** (Seed 42: 54.87%, Seed 123: 50.96%, Seed 456: 49.33%).

### Evidence C: Per-Class Precision and Recall Breakdown
* In the $lr = 2\times 10^{-5}$ configuration, the model heavily favored the majority Negative class (Recall: 0.88), while the minority Neutral class suffered severe false negative suppression (Recall: ~0.36 - 0.42), collapsing unweighted Macro F1.
* In the $lr = 2\times 10^{-4}$ configuration, Neutral Recall doubled to **50.00% - 57.28%**, lifting Macro F1 above 73%.

---

## 3. Root Cause Hypotheses

### Hypothesis 1: Learning Rate Deficit for LoRA Low-Rank Adapters (Primary Cause)
* **Mechanics:** Full fine-tuning of BERT conventionally uses $lr \in [2\times 10^{-5}, 3\times 10^{-5}]$ because larger updates disrupt pretrained representations across all 110M weights. In contrast, LoRA completely freezes the pretrained backbone (110.56M weights) and updates only low-rank decomposition matrices $W = W_0 + \frac{\alpha}{r}BA$. Matrix $B$ is initialized to zero and $A$ to Gaussian random values.
* **Failure Mode:** At $lr = 2\times 10^{-5}$ over only 5 epochs (1,950 total training steps), gradient step sizes are an order of magnitude too small. The adapter cannot overcome the unweighted Cross-Entropy loss slope towards the majority class before training terminates or early stopping triggers.
* **Literature Alignment:** Hu et al. (2021) and standard PEFT best practices recommend learning rates of $1\times 10^{-4}$ to $5\times 10^{-4}$ for LoRA sequence classification.

### Hypothesis 2: Text Representation Discrepancy
* **Mechanics:** The 73.45% legacy run was trained on `text_with_emoticon` (`data_preprocessed_with_emoticon.csv`), where emoticons were converted to explicit textual tokens (e.g. `[senang]`, `[sedih]`). The recent baseline evaluated on `processed_text_v2` (`banjir_processed_v2.csv`), where formal colloquial normalization was applied but emoticons were not converted to Indonesian emotion lexicon tokens.
* **Failure Mode:** In Indonesian social media tweets about disasters, emoticons and punctuation carry disproportionate sentiment signal. Removing or ignoring them deprives the attention heads of vital contextual cues.

### Hypothesis 3: Dropout & Regularization Bottlenecks
* **Mechanics:** The default configuration used `dropout = 0.1`. In Trial 4 of the legacy search, `dropout = 0.3` was selected as optimal. In noisy, short-text tweet classification, higher adapter dropout prevents LoRA from memorizing frequent disaster hashtags (e.g. `#banjirdkijakarta`) and forces robust semantic generalisation.

### Hypothesis 4: Underfitting vs Overfitting Assessment
* **Evidence:** The training loss across 5 epochs in the $lr=2\times 10^{-5}$ runs decreased modestly from 0.85 to 0.68, while validation loss hovered between 0.65 and 0.70.
* **Diagnosis:** The failure is definitively **underfitting**, not overfitting. Training did not plateau due to capacity exhaustion, but due to insufficient gradient velocity.

---

## 4. Recommended Empirical Validation Experiments

| Experiment ID | Focus | Configuration | Primary Measurement | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-DIAG-01** | **Learning Rate Sweep** | $lr \in [2\times 10^{-5}, 5\times 10^{-5}, 1\times 10^{-4}, 2\times 10^{-4}, 3\times 10^{-4}]$ ($r=16, \alpha=32$) | Validation Macro F1 | Macro F1 will correlate directly with $lr$, peaking at $\sim 2\times 10^{-4}$. |
| **EXP-DIAG-02** | **Representation Ablation** | Compare `clean_text` vs `processed_text_v2` vs `text_with_emoticon` at $lr=2\times 10^{-4}$ | Test Macro F1 & Neutral Recall | Quantifies the exact contribution of emoticon lexicon translation. |
| **EXP-DIAG-03** | **LoRA Capacity & Target** | Compare `['query', 'value']` vs `['query', 'key', 'value', 'dense']` at $r=8$ vs $r=16$ | Parameter count vs Macro F1 | Determines whether attention-only adaptation is a representational bottleneck. |
| **EXP-DIAG-04** | **Warmup & Scheduler** | Cosine Annealing vs Linear Decay with 10% Warmup | Stability & Convergence Speed | Identifies smoother convergence dynamics for 3-seed reproducibility. |
