# Task 02 — LLM Completion

## Goal

Reconstruct truncated social media text while preserving meaning.

## Input

`Data/raw/banjir.csv`

Use only these columns:

* text
* sentimen
* label

## Detect truncation

Process only rows containing one of:

* "Tampilkan lebih banyak"
* "View a thread"
* "Lihat selengkapnya"
* text ending with "..."

Rows without these patterns must remain unchanged.

## Rules

* Preserve sentiment.
* Preserve names, locations, numbers, and dates.
* Remove UI artifacts.
* Complete only the unfinished sentence.
* Do not perform stemming.
* Do not normalize slang.

## Output

Create:

`Data/interim/llm_completed.csv`

with new columns:

* llm_completed_text
* llm_status (`completed` or `unchanged`)

## Acceptance Criteria

* Row count unchanged.
* `label` unchanged.
* `sentimen` unchanged.
* Only truncated rows differ from the original.
