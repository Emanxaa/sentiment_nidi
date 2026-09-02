# Comprehensive Evaluation & Synthesis Report - Task 08
**Thesis Project: Sentiment Analysis of Disaster Tweets using LSTM, BiLSTM, and IndoBERTweet-LoRA**

Generated at: `2026-09-02 05:09:14`  
Test Evaluation Cohort: `n = 1,730` test samples (Stratified Split Seed 42)  
Unified Results File: [`experiments/results.csv`](../experiments/results.csv)  

---

## 1. Master Performance Comparison Table

| Model Architecture | Feature Representation | Test Accuracy | Macro F1-Score | Recall Netral | F1 Netral |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (L2)** | TF-IDF (1-2) | 76.30% | **0.6761** | 31.1% | 0.4322 |
| **Linear SVM (LinearSVC)** | TF-IDF (1-2) | 75.78% | **0.6878** | 40.4% | 0.4756 |
| **SGD Classifier (Log Loss)** | TF-IDF (1-2) | 75.95% | **0.6661** | 28.5% | 0.4057 |
| **Multinomial Naive Bayes** | TF-IDF (1-2) | 73.18% | **0.6274** | 22.5% | 0.3469 |
| **Random Forest (100 Trees)** | TF-IDF (1-2) | 74.45% | **0.6223** | 18.5% | 0.2995 |
| **Gradient Boosting** | TF-IDF (1-2) | 73.53% | **0.6211** | 20.9% | 0.3142 |
| **LSTM (Baseline)** | Word Embedding + LSTM | 72.66% | **0.6900** | 55.1% | 0.5283 |
| **BiLSTM (Empiris)** | Word Embedding + BiLSTM | 75.26% | **0.6880** | 49.1% | 0.5126 |
| **IndoBERTweet-LoRA (Empiris)** | Transformer + LoRA (r=16) | 78.73% | **0.7345** | 53.6% | 0.5638 |
| **IndoBERTweet-LoRA (Calibrated w=[1,1.5,1])** | Transformer + LoRA + Calibration | 77.46% | **0.7394** | 66.9% | 0.6012 |

---

## 2. Statistical Significance Analysis (McNemar Test)

1. **IndoBERTweet-LoRA vs Baseline LSTM**:
   - $\chi^2 = 38.42, p < 0.0001$ (**Statistically Significant Superiority**).
   - Contextual word representations from Transformer architecture capture discourse syntax and sentiment nuance substantially better than static LSTM embeddings.

2. **IndoBERTweet-LoRA vs Classical SVM (TF-IDF)**:
   - $\chi^2 = 46.18, p < 0.0001$ (**Statistically Significant Superiority**).
   - TF-IDF bag-of-words fails to resolve word order and semantic shifts in informal disaster tweets.

3. **Threshold Calibration Impact ($w=[1.0, 1.5, 1.0]$)**:
   - Successfully lifts Neutral Recall from **53.58% to 66.89%** ($+13.31\%$) with minimal drop in overall accuracy (78.73% $ightarrow$ 77.46%).
   - Boosts Neutral F1-score past the target threshold to **0.6012**.

---

## 3. Error Analysis & Key Takeaways

1. **Informative Disaster Tweet Ambiguity**:
   - The primary source of classification error across all models stems from informative, factual disaster updates (e.g. reporting water gauge heights or bridge logistics) being misclassified as negative because they contain the trigger word *"banjir"*.
2. **Threshold Calibration Efficiency**:
   - Post-hoc probability threshold tuning effectively remedies the minority class imbalance penalty without the negative side effects of loss distortion or synthetic text noise.
3. **Data-Centric Preprocessing Impact**:
   - Conditional LLM sentence reconstruction (Task 02), deterministic regex refinement (Task 03), and lexicon slang normalization (Task 04) created a reliable dataset foundation that enabled peak model stability.

---

## 4. Pipeline Execution Summary (Tasks 01–08 Complete)

* **Task 01**: Data Audit & Candidate Detection — `Data/interim/audit.csv`
* **Task 02**: Conditional LLM Reconstruction — `Data/interim/llm_completed.csv`
* **Task 03**: Regex Refinement — `Data/interim/regex_clean.csv`
* **Task 04**: Kamus Alay Normalization — `Data/processed/banjir_processed_v2.csv`
* **Task 05**: Emoticon Handling & Dual Pipeline — `Data/processed/data_preprocessed_v2.csv`
* **Task 06**: Stratified Train/Val/Test Split — `Data/processed/split_data_v2.pkl`
* **Task 07**: Model Training & Benchmarking — `reports/benchmark_metrics.csv`
* **Task 08**: Synthesis & Statistical Significance — `reports/task08_synthesis_report.md`
