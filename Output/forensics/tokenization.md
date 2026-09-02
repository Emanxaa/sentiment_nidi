# Milestone B2.5 — Phase 4: Tokenization Forensic Report

**Generated on:** `2026-09-02`  
**Focus:** Tokenizer vocabulary, sequence padding, truncation mechanisms, and input representations.

---

## 1. Tokenizer Specification

| Property | Legacy Notebook (`04_model_indobertweet_lora.ipynb`) | Current Pipeline (`utils/bert_data.py`) | Discrepancy Severity |
| :--- | :--- | :--- | :---: |
| **Tokenizer Name** | `indolem/indobertweet-base-uncased` | `indolem/indobertweet-base-uncased` | Identical |
| **Vocabulary Size** | `31,923` | `31,923` | Identical |
| **Special Tokens** | `[CLS], [SEP], [PAD], [UNK], [MASK]` | `[CLS], [SEP], [PAD], [UNK], [MASK]` | Identical |
| **Max Sequence Length** | `128` | `128` | Identical |
| **Truncation** | `truncation=True` | `truncation=True` | Identical |
| **Padding Strategy** | `padding=True` (Dynamic batch padding) | `padding="max_length"` (Static 128 padding) | Moderate |
| **Input Text Stream** | `text_bert` | `processed_text_v2` / `clean_text` | **CRITICAL** |

---

## 2. The Critical Text Stream Discovery

1. **The Historical Experiment did NOT train on `text_with_emoticon` directly!**
   * Inspection of `legacy_notebooks/01_preprocessing.ipynb` (Cell 17 & 20) revealed:
     ```python
     data_new["clean_text_lstm"] = data_new["text_with_emoticon"].apply(preprocess_lstm)
     data_new["text_bert"] = data_new["text_with_emoticon"].apply(preprocess_bert)
     X_train_bert = df_train["text_bert"].reset_index(drop=True)
     ```
   * The historical model was trained on **`text_bert`**, which was derived by applying `preprocess_bert(text)` on `text_with_emoticon`.
2. **Differences between `text_bert` and `text_with_emoticon`:**
   * Mentions (`@username`) and URLs (`http...`) were stripped.
   * Hashtag symbols (`#word`) were stripped to plain words (`word`).
   * Colons (`:`) were converted to spaces.
   * **52.0% of the dataset rows differ between `text_with_emoticon` and `text_bert`!**
3. **Differences between `text_bert` and `processed_text_v2`:**
   * In `text_bert`, raw Twitter engagement strings (e.g. `... 4 10 790`) and translated emoji strings (`[senang]`, `[sedih]`) were preserved at the end of tweets.
   * In `processed_text_v2`, engagement strings were cleaned and sentences were normalized to formal grammatical Indonesian with LLM completion.
