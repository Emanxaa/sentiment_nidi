"""
Regex Refinement Utility
========================
Task 03 — Thesis-LSTM-IndoBERT
Applies deterministic regex cleaning on `llm_completed_text` from Task 02.
Produces `regex_text` column while strictly preserving original rows, labels, and order.
"""

from __future__ import annotations

import argparse
import html
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def clean_text_regex(text: str) -> str:
    """
    Apply standard deterministic regex refinement on input text.
    1. Unicode normalization (NFKC)
    2. Unescape HTML entities & remove HTML tags
    3. Remove URLs
    4. Remove user mentions (@username)
    5. Remove UI truncation phrases
    6. Remove engagement metric patterns
    7. Convert hashtags to plain words (#topic -> topic)
    8. Clean redundant symbols & normalize whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode normalization (NFKC)
    s = unicodedata.normalize("NFKC", text)

    # 2. HTML unescape and tag removal
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)

    # 3. Remove URLs
    s = re.sub(r"(?i)\b(?:https?://|www\.)\S+", " ", s)
    s = re.sub(r"(?i)\b(?:t\.co|bit\.ly)/\S+", " ", s)

    # 4. Remove mentions (@username)
    s = re.sub(r"@[A-Za-z0-9_]+", " ", s)

    # 5. Remove UI artifacts
    s = re.sub(r"(?i)\b(?:Tampilkan lebih banyak|View a thread|Lihat selengkapnya)\b", " ", s)

    # 6. Remove engagement metrics (e.g. '2 rb', '35 rb', '1 jt', '500 ribu', '1.2 juta')
    s = re.sub(r"(?i)\b\d+(?:[.,]\d+)?\s*(?:rb|jt|ribu|juta|k|m)\b", " ", s)
    # Remove trailing scraped standalone engagement numbers at the very end of line
    s = re.sub(r"(?:\s+\d+)+\s*$", " ", s)

    # 7. Preserve hashtag words (#prayforaceh -> prayforaceh)
    s = re.sub(r"#([A-Za-z0-9_]+)", r"\1", s)

    # 8. Clean residual zero-width or formatting control chars
    s = re.sub(r"[\u200B-\u200F\u2060-\u206F\uFEFF\uFE0F]", "", s)

    # 9. Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def process_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    report_path: str | Path,
    input_col: str = "llm_completed_text",
    output_col: str = "regex_text"
) -> pd.DataFrame:
    """
    Execute full Task 03 regex refinement pipeline.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    report_path = Path(report_path)

    print(f"[1/4] Reading input dataset from: {input_path}")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"      Loaded {total_rows:,} rows.")

    if input_col not in df.columns:
        # Fallback to 'text' if llm_completed_text is missing
        if "text" in df.columns:
            print(f"      Warning: '{input_col}' column not found. Using 'text' column instead.")
            input_col = "text"
        else:
            raise ValueError(f"Required input column '{input_col}' not found.")

    print(f"[2/4] Applying vectorized regex refinement on '{input_col}'...")
    df_out = df.copy()
    df_out[output_col] = df_out[input_col].fillna("").astype(str).apply(clean_text_regex)

    # Check validation
    print("[3/4] Validating outputs and integrity...")
    assert len(df_out) == total_rows, f"Row count changed: {len(df_out)} vs {total_rows}"
    assert (df_out.index == df.index).all(), "Dataframe index modified!"
    
    # Check that labels and sentiments were preserved
    for col in ["label", "sentimen"]:
        if col in df.columns:
            assert (df[col].fillna("__NA__") == df_out[col].fillna("__NA__")).all(), f"Column '{col}' was modified!"

    # Calculate statistics
    urls_left = df_out[output_col].str.contains(r"(?i)https?://|www\.", regex=True).sum()
    mentions_left = df_out[output_col].str.contains(r"@[A-Za-z0-9_]+", regex=True).sum()
    hashtags_left = df_out[output_col].str.contains(r"#[A-Za-z0-9_]+", regex=True).sum()
    changed_count = (df_out[output_col] != df_out[input_col]).sum()

    print(f"      - Changed rows after regex refinement: {changed_count:,} / {total_rows:,} ({changed_count/total_rows*100:.2f}%)")
    print(f"      - Residual URLs: {urls_left}, Residual Mentions: {mentions_left}, Residual Hashtags: {hashtags_left}")

    print(f"[4/4] Saving deliverables...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(output_path, index=False)
    print(f"      - Saved refined dataset to: {output_path}")

    # Generate Markdown Report
    sample_diffs = df_out[df_out[output_col] != df_out[input_col]].head(5)
    example_mds = []
    for i, (_, r) in enumerate(sample_diffs.iterrows(), 1):
        example_mds.append(
            f"### Example {i}\n\n**Before (`{input_col}`):**\n```text\n{r[input_col]}\n```\n\n**After (`{output_col}`):**\n```text\n{r[output_col]}\n```"
        )
    examples_text = "\n\n".join(example_mds)

    report_content = f"""# Regex Refinement Report - Task 03

Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  
Input: `{input_path}`  
Output: `{output_path}`  

---

## 1. Summary Statistics

* **Total Rows:** {total_rows:,}
* **Rows Modified by Regex Cleaning:** {changed_count:,} ({changed_count/total_rows*100:.2f}%)
* **Unmodified Rows:** {total_rows - changed_count:,} ({(total_rows - changed_count)/total_rows*100:.2f}%)
* **Residual URLs:** {urls_left}
* **Residual User Mentions (`@username`):** {mentions_left}
* **Residual Hashtag Symbols (`#word`):** {hashtags_left} (Words converted to plain text)

---

## 2. Cleaning Rules Applied

1. **Unicode Normalization**: NFKC normalization applied.
2. **URL Removal**: `http://`, `https://`, `www.`, `t.co/`, `bit.ly/` stripped.
3. **Mention Removal**: `@username` stripped.
4. **Engagement Metric Removal**: `2 rb`, `35 rb`, `1 jt`, `ribu`, `juta` stripped.
5. **UI Artifact Removal**: Residual truncation phrases and HTML entities stripped.
6. **Hashtag Preservation**: `#prayforaceh` converted to `prayforaceh`.
7. **Whitespace Normalization**: Whitespace collapsed and trimmed.

---

## 3. Before / After Examples

{examples_text}

---

## 4. Integrity Verification

* **Row Count Preserved:** Yes ({total_rows:,} rows).
* **Labels and Sentiments Unchanged:** Yes.
* **No Stemming or Slang Normalization Applied:** Yes (Preserved for Task 04).
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"      - Saved report to: {report_path}")

    return df_out


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    default_input = project_root / "Data" / "interim" / "llm_completed.csv"
    default_output = project_root / "Data" / "interim" / "regex_clean.csv"
    default_report = project_root / "Data" / "interim" / "regex_refinement_report.md"

    parser = argparse.ArgumentParser(description="Regex Refinement Pipeline (Task 03)")
    parser.add_argument("--input", type=str, default=str(default_input), help="Path to input llm_completed.csv")
    parser.add_argument("--output", type=str, default=str(default_output), help="Path to output regex_clean.csv")
    parser.add_argument("--report", type=str, default=str(default_report), help="Path to output markdown report")
    parser.add_argument("--input-col", type=str, default="llm_completed_text", help="Column name to clean")
    parser.add_argument("--output-col", type=str, default="regex_text", help="New column name for cleaned text")

    args = parser.parse_args()

    process_dataset(
        input_csv=args.input,
        output_csv=args.output,
        report_path=args.report,
        input_col=args.input_col,
        output_col=args.output_col,
    )


if __name__ == "__main__":
    main()
