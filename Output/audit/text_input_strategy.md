# Text Input Strategy Audit & Representation Analysis

**Document Version:** 1.0.0  
**Date:** 2026-09-02  
**Role:** ML Research Engineer  
**Scope:** Comparative Analysis of Available Preprocessing Representations for IndoBERTweet-LoRA vs LSTM  

---

## 1. Executive Summary

A foundational tenet of modern Natural Language Processing is that **Transformer architectures (BERT)** and **Recurrent Neural Networks (LSTM)** exhibit fundamentally divergent requirements regarding text preprocessing:
* **Recurrent Architectures (LSTM):** Bound by fixed word-level embedding matrices ($V \times D$). They benefit from aggressive lexical reduction (lowercasing, punctuation removal, stemming, stopword stripping) to compress the vocabulary size and mitigate Out-Of-Vocabulary (OOV) sparsity.
* **Pretrained Bidirectional Transformers (IndoBERTweet):** Bound by subword tokenization (Byte-Pair Encoding / WordPiece) trained on raw unstemmed natural language. They rely heavily on **syntactic markers, casing (acronyms, shout-caps), punctuation boundaries, and emoji semantics** to calculate multi-head attention weights across bidirectional contexts. Aggressive text stripping severely degrades transformer representation quality.

This document systematically audits every available text column in `Data/processed/banjir_processed_v2.csv` ($n=8,648$), details what information is preserved versus discarded, evaluates model suitability, and establishes an evidence-based input priority order for future experiments.

---

## 2. Granular Column-by-Column Audit

The canonical dataset `Data/processed/banjir_processed_v2.csv` contains multiple historical and refined representation stages:

```
+----------------------------------------------------------------------------------------------------+
|                                    DATA TRANSFORMATION PIPELINE                                    |
| [1. Raw Tweet: text]                                                                               |
|         |                                                                                          |
|         +---> [2. Regex & Informal Cleaning: clean_text]                                           |
|         |                                                                                          |
|         +---> [3. Sastrawi Stemming: processed_text]                                               |
|         |                                                                                          |
|         +---> [4. LLM Completion + Slang Lexicon Normalization: processed_text_v2]                 |
+----------------------------------------------------------------------------------------------------+
```

### 1. `text` (Raw Social Media Tweet)
* **Sample (Row 0):** `"Polisi : Sulit menemukan siapa pemilik kayu yg menyebabkan banjir Sumatra akhir 2025. Rakyat : Polisi Goblok, Percuma sekolah, Itu di BPN kan ada siapa pemilik lahan HPH..."`
* **Information Preserved:** Original grammatical syntax, casing, dialog colons, full punctuation (`.`, `,`, `!`, `?`), URLs, mentions (`@user`), hashtags (`#banjir`), emojis, Indonesian colloquial slang (`yg`, `gak`, `krn`), numbers (`2025`).
* **Information Removed:** None (raw scraped stream).
* **Suitability for IndoBERTweet:** **Moderate.** IndoBERTweet was pretrained on Indonesian Twitter data and can tokenize informal text. However, uncurated URLs, dangling mentions, and scraped truncation artifacts add sequence noise without sentiment value.
* **Suitability for LSTM:** **Very Poor.** Causes massive vocabulary inflation, heavy OOV occurrences, and diluted embedding representations.

---

### 2. `clean_text` (Traditional NLP Preprocessing)
* **Sample (Row 0):** `"polisi sulit menemukan siapa pemilik kayu yg menyebabkan banjir sumatra akhir rakyat polisi goblok percuma sekolah itu di bpn kan ada siapa pemilik lahan hph..."`
* **Information Preserved:** Base Indonesian word sequences, raw colloquial tokens (`yg`, `gak`, `krn`).
* **Information Removed:** All uppercase casing, all punctuation (colons, commas, periods, exclamation marks, question marks), all numbers, user mentions, URLs, hashtags, emojis/emoticons.
* **Suitability for IndoBERTweet:** **Suboptimal / Degraded.** 
  * *Punctuation Loss:* Stripping punctuation destroys sentence and clause boundaries, flattening the attention mechanism's ability to isolate conversational turns or emphatic sentiment shifts.
  * *Casing Loss:* Stripping uppercase removes anger/urgency indicators (e.g. `"GOBLOK"`, `"PERCUMA"`, `"BPN"` vs `"bpn"`).
  * *Emoji Loss:* Disaster sentiment on social media frequently relies on emotion icons to convey sadness, frustration, or relief.
* **Suitability for LSTM:** **High.** Provides a clean, vocabulary-bounded sequence ideal for fixed integer tokenizers and word embedding lookups.

---

### 3. `processed_text` (Stemmed Representation via Sastrawi)
* **Sample (Row 0):** `"polisi sulit temu siapa milik kayu sebab banjir sumatra akhir rakyat polisi goblok percuma sekolah bpn kan siapa milik lahan hph presiden tri taipan..."`
* **Information Preserved:** Root morphological base forms (`menemukan` $\rightarrow$ `temu`, `pemilik` $\rightarrow$ `milik`, `menyebabkan` $\rightarrow$ `sebab`).
* **Information Removed:** All affixes (prefixes `me-`, `di-`, `ber-`, `pe-`; suffixes `-kan`, `-an`, `-i`), casing, punctuation, stopwords, sentiment markers.
* **Suitability for IndoBERTweet:** **Extremely Destructive.** IndoBERTweet was pretrained on standard natural language morphology. Root-only sequences force the WordPiece tokenizer to segment distorted roots into fragmented subword artifacts, completely destroying the semantic representations in pretrained Transformer layers.
* **Suitability for LSTM:** **Moderate to High.** Helps compress lexical variations into shared indices for simple recurrent models.

---

### 4. `processed_text_v2` (LLM-Completed & Slang-Normalized)
* **Sample (Row 0):** `"Polisi : Sulit menemukan siapa pemilik kayu yang menyebabkan banjir Sumatra akhir 2025. Rakyat : Polisi Goblok, Percuma sekolah, Itu di BPN kan ada siapa pemilik lahan HPH. Presiden, Mentri, Taipan Pemegang HPH, Pengusaha Sawit. Memang enggak niat ungkap, karena kelas Paus semua."`
* **Information Preserved:** 
  * Full syntactic casing (distinguishes proper nouns, acronyms like `BPN`/`HPH`, and emphatic expressions).
  * Structural punctuation (dialog colons, commas, periods, sentence boundaries).
  * Complete semantic clauses restored via contextual LLM inference where tweets were cut off.
  * Numbers and temporal anchors (`2025`).
* **Information Normalized / Removed:** 
  * Colloquial and alay slang converted to standard formal Indonesian equivalents (`yg` $\rightarrow$ `yang`, `gak` $\rightarrow$ `enggak`, `krn` $\rightarrow$ `karena`).
  * Noisy URL links and user handles cleanly eliminated without distorting sentence flow.
* **Suitability for IndoBERTweet:** **Optimal / Highest.** 
  * Matches the linguistic expectations of the pretrained Transformer encoder.
  * Preserved punctuation allows the multi-head self-attention layers to identify sentence structure.
  * Standardized spelling maps directly to existing high-frequency tokens in IndoBERTweet's 30,000 WordPiece vocabulary, drastically reducing rare subword fragmentation.
* **Suitability for LSTM:** **Moderate to High.** Standardized words improve word-embedding hits, though casing slightly increases vocabulary size compared to `clean_text`.

---

## 3. Comparative Summary Matrix

| Evaluation Dimension | `text` (Raw) | `clean_text` | `processed_text` | `processed_text_v2` |
| :--- | :---: | :---: | :---: | :---: |
| **Casing Preserved** | Yes | No | No | **Yes** |
| **Punctuation Preserved** | Yes | No | No | **Yes** |
| **Slang Normalized** | No | No | No | **Yes (Formal)** |
| **Stemming Applied** | No | No | Yes (Sastrawi) | **No** |
| **LLM Context Restored**| No | No | No | **Yes** |
| **Subword Token Fragmentation** | Moderate | High (informal) | Very High (distorted) | **Lowest (Optimal)** |
| **Suitability for BERT** | 6 / 10 | 5 / 10 | 2 / 10 | **9.5 / 10** |
| **Suitability for LSTM** | 4 / 10 | **9 / 10** | 7.5 / 10 | 8 / 10 |

---

## 4. Evidence-Based Priority Order for Future Experiments

Based on architectural mechanics and observed empirical evidence, text inputs for future IndoBERTweet-LoRA experiments must follow this strict priority order:

1. **Priority 1 (Primary Benchmark Stream): `processed_text_v2`**
   * *Rationale:* Combines noise removal with full syntactic structure, casing, and standard vocabulary mapping. Directly comparable to the completed LSTM baseline suite.
2. **Priority 2 (Ablation Benchmark Stream): `clean_text`**
   * *Rationale:* Evaluated in Milestone B2/Stage A specifically to prove the thesis hypothesis that Transformers lose critical attention context when punctuation and casing are stripped.
3. **Priority 3 (Exploratory / Supplemental): `text_with_emoticon`**
   * *Rationale:* Legacy Trial 4 achieved 73.45% Macro F1 with explicit emotion lexicon translation (e.g. `[senang]`, `[sedih]`). Should be tested if minority Neutral/Positive classes require additional semantic boosting.
4. **Prohibited for BERT: `processed_text` (Stemmed)**
   * *Rule:* Never feed stemmed text into IndoBERTweet; subword fragmentation destroys pretrained transformer embeddings.
