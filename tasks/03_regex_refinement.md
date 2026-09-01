# Task 03 — Regex Refinement

## Goal

Apply deterministic regex cleaning after LLM completion.

## Input

`Data/interim/llm_completed.csv`

Column:

* llm_completed_text

## Apply

1. Unicode normalization (NFKC).
2. Remove URLs.
3. Remove mentions (`@username`).
4. Remove engagement metrics (`2 rb`, `35 rb`, dll.).
5. Remove UI artifacts that remain.
6. Preserve hashtag words (`#prayforaceh` → `prayforaceh`).
7. Normalize whitespace.

## Do Not

* Stem words.
* Remove meaningful words.
* Change labels.

## Output

`Data/interim/regex_clean.csv`

Add column:

* regex_text

## Acceptance Criteria

* Same row count.
* `label` unchanged.
* `sentimen` unchanged.
