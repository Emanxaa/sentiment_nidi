"""
Kamus Alay Normalization Utility
=================================
Task 04 — Thesis-LSTM-IndoBERT
Normalizes Indonesian colloquial / slang words using lexicon lookup.
Preserves tokens not found in the lexicon. Strictly non-destructive for labels.
"""

from __future__ import annotations

import argparse
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


def load_normalization_dictionary(
    lexicon_path: str | Path,
    extra_dict_path: str | Path | None = None
) -> Dict[str, str]:
    """
    Load slang-to-formal mapping dictionary from CSV and optional python dict.
    """
    norm_dict: Dict[str, str] = {}
    lexicon_file = Path(lexicon_path)

    if lexicon_file.exists():
        df_lex = pd.read_csv(lexicon_file)
        if "slang" in df_lex.columns and "formal" in df_lex.columns:
            for _, row in df_lex.iterrows():
                slang = str(row["slang"]).strip().lower()
                formal = str(row["formal"]).strip().lower()
                if slang and formal and slang != "nan" and formal != "nan":
                    norm_dict[slang] = formal

    # Merge additional project dictionary if exists
    if extra_dict_path:
        extra_file = Path(extra_dict_path)
        if extra_file.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("extra_norm", extra_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "normalisasi_dict"):
                    for k, v in module.normalisasi_dict.items():
                        norm_dict[k.lower().strip()] = v.lower().strip()

    print(f"[Dictionary] Loaded {len(norm_dict):,} slang-to-formal mappings.")
    return norm_dict


def normalize_sentence(text: str, norm_dict: Dict[str, str]) -> str:
    """
    Normalize tokens in a sentence using dictionary lookup.
    Handles tokenization while preserving punctuation attachments.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    tokens = text.split()
    normalized_tokens = []

    for token in tokens:
        # Separate leading/trailing punctuation if attached
        prefix_punct = ""
        suffix_punct = ""
        core_token = token

        # Strip leading punctuation
        while core_token and core_token[0] in string.punctuation:
            prefix_punct += core_token[0]
            core_token = core_token[1:]

        # Strip trailing punctuation
        while core_token and core_token[-1] in string.punctuation:
            suffix_punct = core_token[-1] + suffix_punct
            core_token = core_token[:-1]

        lower_token = core_token.lower()
        if lower_token in norm_dict:
            replaced_token = norm_dict[lower_token]
            # Match capitalization if original was capitalized
            if core_token.istitle():
                replaced_token = replaced_token.capitalize()
            elif core_token.isupper() and len(core_token) > 1:
                replaced_token = replaced_token.upper()
            normalized_tokens.append(f"{prefix_punct}{replaced_token}{suffix_punct}")
        else:
            normalized_tokens.append(token)

    return " ".join(normalized_tokens)


def process_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    report_path: str | Path,
    lexicon_csv: str | Path,
    extra_dict_path: str | Path | None = None,
    input_col: str = "regex_text",
    output_col: str = "processed_text_v2"
) -> pd.DataFrame:
    """
    Execute full Task 04 alay normalization pipeline.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    report_path = Path(report_path)

    print(f"[1/4] Reading input dataset from: {input_path}")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"      Loaded {total_rows:,} rows.")

    if input_col not in df.columns:
        if "llm_completed_text" in df.columns:
            input_col = "llm_completed_text"
        else:
            input_col = "text"
        print(f"      Warning: Using column '{input_col}' as input.")

    print(f"[2/4] Loading lexicon dictionaries...")
    norm_dict = load_normalization_dictionary(lexicon_csv, extra_dict_path)

    print(f"[3/4] Applying dictionary normalization on '{input_col}'...")
    df_out = df.copy()
    df_out[output_col] = df_out[input_col].fillna("").astype(str).apply(
        lambda t: normalize_sentence(t, norm_dict)
    )

    # Validations
    assert len(df_out) == total_rows, "Row count changed!"
    assert (df_out.index == df.index).all(), "Index modified!"
    for col in ["label", "sentimen"]:
        if col in df.columns:
            assert (df[col].fillna("__NA__") == df_out[col].fillna("__NA__")).all(), f"Column '{col}' changed!"

    changed_count = (df_out[output_col] != df_out[input_col]).sum()
    print(f"      - Rows modified by slang normalization: {changed_count:,} / {total_rows:,} ({changed_count/total_rows*100:.2f}%)")

    print(f"[4/4] Saving deliverables...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(output_path, index=False)
    print(f"      - Saved processed dataset to: {output_path}")

    # Generate Report
    sample_diffs = df_out[df_out[output_col] != df_out[input_col]].head(5)
    example_mds = []
    for i, (_, r) in enumerate(sample_diffs.iterrows(), 1):
        example_mds.append(
            f"### Example {i}\n\n**Before (`{input_col}`):**\n```text\n{r[input_col]}\n```\n\n**After (`{output_col}`):**\n```text\n{r[output_col]}\n```"
        )
    examples_text = "\n\n".join(example_mds)

    report_content = f"""# Kamus Alay Normalization Report - Task 04

Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  
Input: `{input_path}`  
Output: `{output_path}`  
Lexicon Dictionary: `{lexicon_csv}` ({len(norm_dict):,} entries)  

---

## 1. Summary Statistics

* **Total Rows:** {total_rows:,}
* **Rows Normalized:** {changed_count:,} ({changed_count/total_rows*100:.2f}%)
* **Unmodified Rows:** {total_rows - changed_count:,} ({(total_rows - changed_count)/total_rows*100:.2f}%)
* **Lookup Method:** Pure deterministic dictionary replacement (no LLM hallucination).

---

## 2. Before / After Normalization Examples

{examples_text}

---

## 3. Integrity Verification

* **Row Count Preserved:** Yes ({total_rows:,} rows).
* **Labels and Sentiments Unchanged:** Yes.
* **Non-lexicon tokens preserved:** Yes.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"      - Saved report to: {report_path}")

    return df_out


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    default_input = project_root / "Data" / "interim" / "regex_clean.csv"
    default_output = project_root / "Data" / "processed" / "banjir_processed_v2.csv"
    default_report = project_root / "Data" / "processed" / "alay_normalization_report.md"
    default_lexicon = project_root / "kamus" / "colloquial-indonesian-lexicon.csv"
    default_extra = project_root / "Preprocessing" / "normalisasi_dict.py"

    parser = argparse.ArgumentParser(description="Kamus Alay Normalization Pipeline (Task 04)")
    parser.add_argument("--input", type=str, default=str(default_input), help="Path to input regex_clean.csv")
    parser.add_argument("--output", type=str, default=str(default_output), help="Path to output banjir_processed_v2.csv")
    parser.add_argument("--report", type=str, default=str(default_report), help="Path to output report")
    parser.add_argument("--lexicon", type=str, default=str(default_lexicon), help="Path to colloquial lexicon CSV")
    parser.add_argument("--extra-dict", type=str, default=str(default_extra), help="Path to extra normalisasi_dict.py")

    args = parser.parse_args()

    process_dataset(
        input_csv=args.input,
        output_csv=args.output,
        report_path=args.report,
        lexicon_csv=args.lexicon,
        extra_dict_path=args.extra_dict,
    )


if __name__ == "__main__":
    main()
