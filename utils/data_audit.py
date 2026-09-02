"""
Data Audit & Candidate Detection Utility
=========================================
Audits the raw dataset and detects candidate patterns requiring downstream processing
(such as LLM completion, regex cleaning, URL/mention removal, and unicode normalization).

Strictly non-destructive: does not alter text, labels, or row count.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


# -----------------------------------------------------------------------------
# Detection Helper Functions (Vectorized Pandas)
# -----------------------------------------------------------------------------

def detect_truncation(series: pd.Series) -> pd.Series:
    """
    Detect text truncation patterns such as UI snippets or ending ellipsis.
    Matches:
      - 'Tampilkan lebih banyak'
      - 'View a thread'
      - 'Lihat selengkapnya'
      - Text ending with '...' or '…' (including before trailing engagement metrics)
    """
    pattern = r"(?i)(?:Tampilkan lebih banyak|View a thread|Lihat selengkapnya|(?:\.\.\.|…)\s*(?:\d+\s*)*$)"
    return series.fillna("").astype(str).str.contains(pattern, regex=True)


def detect_mention(series: pd.Series) -> pd.Series:
    """
    Detect presence of Twitter/social media user mentions (@username).
    """
    pattern = r"@[A-Za-z0-9_]+"
    return series.fillna("").astype(str).str.contains(pattern, regex=True)


def detect_hashtag(series: pd.Series) -> pd.Series:
    """
    Detect presence of hashtags (#word).
    """
    pattern = r"#[A-Za-z0-9_]+"
    return series.fillna("").astype(str).str.contains(pattern, regex=True)


def detect_url(series: pd.Series) -> pd.Series:
    """
    Detect presence of URLs (http://, https://, www., t.co/, bit.ly/, etc.).
    """
    pattern = r"(?i)(?:https?://\S+|www\.\S+|t\.co/\S+|bit\.ly/\S+)"
    return series.fillna("").astype(str).str.contains(pattern, regex=True)


def detect_unicode(series: pd.Series) -> pd.Series:
    """
    Detect unusual Unicode characters such as mojibake ('Â'),
    mathematical bold/italic alphanumeric symbols, zero-width spaces,
    foreign scripts (Arabic, Thai, CJK, Hangul, Lao, Yi), braille, dingbats,
    and phonetic modifier letters.
    """
    pattern = (
        r"[\U0001D400-\U0001D7FF"                  # Mathematical Alphanumeric Symbols
        r"\u0E80-\u0EFF"                             # Lao
        r"\u0E00-\u0E7F"                             # Thai
        r"\u0600-\u06FF"                             # Arabic
        r"\u4E00-\u9FFF"                             # CJK Unified Ideographs
        r"\u3040-\u30FF"                             # Hiragana & Katakana
        r"\u1100-\u11FF\uAC00-\uD7AF"                 # Hangul Jamo & Syllables
        r"\uA490-\uA4CF"                             # Yi Radicals
        r"\u2800-\u28FF"                             # Braille Patterns
        r"\u200B-\u200F\u2060-\u206F\uFEFF\uFE0F"     # Zero-width, directional & format chars
        r"\u0300-\u036F"                             # Combining Diacritical Marks
        r"\u1D00-\u1DBF\u02B0-\u02FF\u2070-\u209F" # Phonetic modifiers & superscripts/subscripts
        r"\uE000-\uF8FF"                             # Private Use Area
        r"\u2460-\u24FF\U0001F100-\U0001F1FF"         # Enclosed Alphanumerics
        r"\u2600-\u27BF"                             # Misc Symbols & Dingbats
        r"]|Â"                                       # Mojibake marker (Â)
    )
    return series.fillna("").astype(str).str.contains(pattern, regex=True)


def detect_engagement(series: pd.Series) -> pd.Series:
    """
    Detect engagement count patterns (e.g. '2 rb', '35 rb', '1 jt', '500 ribu', '1.2 juta').
    """
    pattern = r"(?i)\b\d+(?:[.,]\d+)?\s*(?:rb|jt|ribu|juta|k|m)\b"
    return series.fillna("").astype(str).str.contains(pattern, regex=True)


def detect_html(series: pd.Series) -> pd.Series:
    """
    Detect HTML artifacts, entities, or tags (e.g., '&amp;', '&ndash;', '<br>').
    """
    pattern = r"&[a-zA-Z0-9#]+;|<[^>]+>"
    return series.fillna("").astype(str).str.contains(pattern, regex=True)


# -----------------------------------------------------------------------------
# Audit Pipeline & Candidate Detection
# -----------------------------------------------------------------------------

def apply_candidate_detection(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Apply all 7 candidate detection flags to the dataframe without modifying existing data.
    """
    df_out = df.copy()
    series = df_out[text_column]

    df_out["has_truncation"] = detect_truncation(series)
    df_out["has_mention"] = detect_mention(series)
    df_out["has_hashtag"] = detect_hashtag(series)
    df_out["has_url"] = detect_url(series)
    df_out["has_unicode"] = detect_unicode(series)
    df_out["has_engagement"] = detect_engagement(series)
    df_out["has_html"] = detect_html(series)

    return df_out


def compute_audit_metrics(df_raw: pd.DataFrame, df_audited: pd.DataFrame, text_column: str = "text") -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics for reporting.
    """
    total_rows = len(df_raw)
    total_cols = len(df_raw.columns)
    
    # Missing values and dtypes
    missing_counts = df_raw.isnull().sum().to_dict()
    missing_pcts = (df_raw.isnull().mean() * 100).round(2).to_dict()
    dtypes = {col: str(dtype) for col, dtype in df_raw.dtypes.items()}

    # Duplicate rows on text
    duplicate_count = int(df_raw.duplicated(subset=[text_column]).sum())

    # Label distributions
    label_dist = df_raw["label"].value_counts(dropna=False).to_dict() if "label" in df_raw.columns else {}
    sentimen_dist = df_raw["sentimen"].value_counts(dropna=False).to_dict() if "sentimen" in df_raw.columns else {}

    # Candidate flags summary
    flags = [
        "has_truncation",
        "has_mention",
        "has_hashtag",
        "has_url",
        "has_unicode",
        "has_engagement",
        "has_html",
    ]
    flag_counts = {flag: int(df_audited[flag].sum()) for flag in flags}
    flag_pcts = {flag: round(float(df_audited[flag].mean() * 100), 2) for flag in flags}

    # Combined candidate rows (rows having at least 1 flag)
    any_flag = df_audited[flags].any(axis=1).sum()

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "duplicate_count": duplicate_count,
        "label_dist": label_dist,
        "sentimen_dist": sentimen_dist,
        "flag_counts": flag_counts,
        "flag_pcts": flag_pcts,
        "any_flag_count": int(any_flag),
        "any_flag_pct": round(float(any_flag / total_rows * 100), 2) if total_rows > 0 else 0.0,
    }


def generate_markdown_report(metrics: Dict[str, Any], input_filename: str = "banjir.csv") -> str:
    """
    Generate clean, structured markdown content for audit_report.md.
    """
    total_rows = metrics["total_rows"]
    
    # Missing value & Dtype table
    col_rows = []
    for col, dtype in metrics["dtypes"].items():
        m_count = metrics["missing_counts"].get(col, 0)
        m_pct = metrics["missing_pcts"].get(col, 0.0)
        col_rows.append(f"| `{col}` | `{dtype}` | {m_count:,} | {m_pct:.2f}% |")
    col_table = "\n".join(col_rows)

    # Label distribution table
    label_rows = []
    if metrics["label_dist"]:
        for label_val, count in metrics["label_dist"].items():
            pct = (count / total_rows * 100) if total_rows > 0 else 0.0
            label_rows.append(f"| `{label_val}` | {count:,} | {pct:.2f}% |")
    label_table = "\n".join(label_rows)

    # Sentimen distribution table
    sentimen_rows = []
    if metrics["sentimen_dist"]:
        for sent_val, count in metrics["sentimen_dist"].items():
            pct = (count / total_rows * 100) if total_rows > 0 else 0.0
            sentimen_rows.append(f"| `{sent_val}` | {count:,} | {pct:.2f}% |")
    sentimen_table = "\n".join(sentimen_rows)

    # Summary table for candidate detection flags
    flag_rows = []
    for flag, count in metrics["flag_counts"].items():
        pct = metrics["flag_pcts"].get(flag, 0.0)
        flag_rows.append(f"| `{flag}` | {count:,} | {pct:.2f}% |")
    flag_table = "\n".join(flag_rows)

    report = f"""# Data Audit Report - {input_filename}

Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  
Source: `Data/raw/{input_filename}`  

---

## 1. Dataset Overview (Part A)

* **Total Rows:** {metrics['total_rows']:,}
* **Total Columns:** {metrics['total_cols']:,}
* **Duplicate Rows (based on `text`):** {metrics['duplicate_count']:,}
* **Rows with at least one candidate flag:** {metrics['any_flag_count']:,} ({metrics['any_flag_pct']}%)

### Column Schema & Missing Values

| Column | Data Type | Missing Count | Missing Percentage |
| :--- | :--- | :--- | :--- |
{col_table}

### Label Distribution (`label`)

| Label | Count | Percentage |
| :--- | :--- | :--- |
{label_table}

### Sentiment Distribution (`sentimen`)

| Sentimen | Count | Percentage |
| :--- | :--- | :--- |
{sentimen_table}

---

## 2. Candidate Detection Summary (Part B & C)

| Flag | Count | Percentage |
| :--- | :--- | :--- |
{flag_table}

---

## 3. Flag Details & Downstream Mapping

| Flag | Description / Matching Condition | Downstream Task Target |
| :--- | :--- | :--- |
| `has_truncation` | Contains UI truncation text (*Tampilkan lebih banyak*, *View a thread*, *Lihat selengkapnya*) or ending ellipsis `...` / `…` | **Task 02: LLM Completion** |
| `has_mention` | Contains `@username` handles | **Task 03: Regex Refinement** |
| `has_hashtag` | Contains `#hashtag` topics | **Task 03: Regex Refinement** |
| `has_url` | Contains web links (`http://`, `https://`, `t.co/`, etc.) | **Task 03: Regex Refinement** |
| `has_unicode` | Contains unusual Unicode (mojibake `Â`, math bold symbols, non-Latin scripts, braille) | **Task 03 / Preprocessing** |
| `has_engagement` | Contains engagement count artifacts (e.g. `2 rb`, `35 rb`, `1 jt`) | **Task 03: Regex Refinement** |
| `has_html` | Contains HTML entities/tags (`&amp;`, `&ndash;`, `<br>`) | **Task 03: Regex Refinement** |

---

## 4. Integrity Verification

* Original row count preserved: **Yes** ({metrics['total_rows']:,} rows).
* Original text and labels modified: **No** (Audit and candidate detection performed strictly read-only).
* Output dataset saved to: `Data/interim/audit.csv`.
"""
    return report


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

def run_audit(
    input_path: str | Path,
    output_csv: str | Path,
    output_report: str | Path,
    text_column: str = "text"
) -> None:
    """
    Execute full audit pipeline: read raw data, apply candidate detection,
    compute metrics, save audited CSV, and write markdown report.
    """
    input_path = Path(input_path)
    output_csv = Path(output_csv)
    output_report = Path(output_report)

    if not input_path.exists():
        # Fallback to Data/data_banjir.csv if input_path doesn't exist
        alt_path = input_path.parent.parent / "Data" / "data_banjir.csv"
        if alt_path.exists():
            print(f"Warning: {input_path} not found. Falling back to {alt_path}")
            input_path = alt_path
        else:
            raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[1/4] Reading input dataset from: {input_path}")
    df_raw = pd.read_csv(input_path)
    print(f"      Loaded {len(df_raw):,} rows and {len(df_raw.columns)} columns.")

    print("[2/4] Running candidate detection flags...")
    df_audited = apply_candidate_detection(df_raw, text_column=text_column)

    print("[3/4] Computing audit metrics and generating report...")
    metrics = compute_audit_metrics(df_raw, df_audited, text_column=text_column)
    report_content = generate_markdown_report(metrics, input_filename=input_path.name)

    print(f"[4/4] Saving deliverables...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    df_audited.to_csv(output_csv, index=False)
    print(f"      - Audited dataset saved to: {output_csv} ({len(df_audited):,} rows, {len(df_audited.columns)} cols)")

    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"      - Audit report saved to: {output_report}")

    print("\nAudit completed successfully!")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    default_input = project_root / "Data" / "raw" / "banjir.csv"
    default_output_csv = project_root / "Data" / "interim" / "audit.csv"
    default_output_report = project_root / "Data" / "interim" / "audit_report.md"

    parser = argparse.ArgumentParser(description="Dataset Audit & Candidate Detection Pipeline")
    parser.add_argument("--input", type=str, default=str(default_input), help="Path to raw CSV file")
    parser.add_argument("--output-csv", type=str, default=str(default_output_csv), help="Path to output audited CSV")
    parser.add_argument("--output-report", type=str, default=str(default_output_report), help="Path to output markdown report")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name to audit")

    args = parser.parse_args()

    run_audit(
        input_path=args.input,
        output_csv=args.output_csv,
        output_report=args.output_report,
        text_column=args.text_col,
    )


if __name__ == "__main__":
    main()
