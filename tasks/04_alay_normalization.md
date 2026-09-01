# Task 05 — Kamus Alay Normalization

## Goal

Normalize Indonesian colloquial words using dictionary lookup.

## Input

`Data/interim/regex_clean.csv`

Dictionary:

`kamus/colloquial-indonesian-lexicon.csv`

## Method

* Tokenize by whitespace.
* Replace tokens only if they exist in the dictionary.
* Leave unknown tokens unchanged.

## Output

`Data/processed/banjir_processed_v2.csv`

Add column:

* processed_text_v2

## Acceptance Criteria

* Row count unchanged.
* Labels unchanged.
* Dictionary lookup only (no LLM).
