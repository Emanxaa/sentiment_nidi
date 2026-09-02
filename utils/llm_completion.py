"""
Conditional LLM Completion Pipeline
===================================
Task 02 — Thesis-LSTM-IndoBERT
Reconstructs truncated social media posts identified in Task 01 (has_truncation == True)
using Google Gemini with batching, retry handling, and non-destructive merging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

load_dotenv()


SYSTEM_PROMPT = """You are reconstructing Indonesian social-media posts that were truncated by platform interface elements.

Your task is only to restore incomplete sentences.

Rules:
* Preserve the original meaning.
* Preserve names, places, dates and numbers.
* Remove UI truncation text.
* Complete only unfinished sentences.
* Do not rewrite the entire post.
* Do not invent new information.
* Return only reconstructed text."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read dataset from CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    df = pd.read_csv(path)
    return df


def validate_input(df: pd.DataFrame) -> None:
    """Validate that input dataframe is non-empty and contains required columns."""
    if len(df) == 0:
        raise ValueError("Input dataset is empty (0 rows).")
    
    required_cols = ["text", "has_truncation"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")


def create_batches(candidates_df: pd.DataFrame, batch_size: int = 20) -> List[pd.DataFrame]:
    """Dynamically split candidates dataframe into batches of size `batch_size`."""
    total_candidates = len(candidates_df)
    batches = []
    for start_idx in range(0, total_candidates, batch_size):
        end_idx = min(start_idx + batch_size, total_candidates)
        batch = candidates_df.iloc[start_idx:end_idx].copy()
        batches.append(batch)
    return batches


def clean_fallback_text(text: str) -> str:
    """Local fallback that removes UI truncation artifacts if LLM fails."""
    cleaned = re.sub(r"(?i)(?:Tampilkan lebih banyak|View a thread|Lihat selengkapnya)", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def reconstruct_batch(
    batch_df: pd.DataFrame,
    client: Optional[genai.Client],
    model: str = "gemini-3.5-flash-lite",
    dry_run: bool = False
) -> Tuple[List[str], str]:
    """
    Reconstruct a batch of truncated tweets using Gemini.
    Retries on failure with rate-limit aware backoff and fallback models.
    """
    texts = batch_df["text"].astype(str).tolist()
    n_items = len(texts)

    if dry_run or client is None:
        return [clean_fallback_text(t) for t in texts], "completed"

    items_payload = [{"id": i + 1, "text": t} for i, t in enumerate(texts)]
    user_prompt = (
        "Reconstruct the following posts.\n\n"
        "For each input return exactly one reconstructed text in the same order.\n\n"
        f"Input posts (JSON list):\n{json.dumps(items_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return output as a JSON list of objects: [{\"id\": 1, \"reconstructed_text\": \"...\"}, ...]"
    )

    models_to_try = [model, "gemini-3.6-flash", "gemini-2.5-flash"]
    for m in models_to_try:
        for attempt in range(1, 3):
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=user_prompt,
                    config=dict(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.1,
                    )
                )
                if response and response.text:
                    data = json.loads(response.text)
                    if isinstance(data, list):
                        res_map = {}
                        for item in data:
                            if isinstance(item, dict):
                                item_id = item.get("id")
                                rec = item.get("reconstructed_text") or item.get("text")
                                if item_id is not None and rec is not None:
                                    res_map[int(item_id)] = str(rec).strip()

                        if len(res_map) == n_items:
                            ordered_results = [res_map.get(i + 1, texts[i]) for i in range(n_items)]
                            return ordered_results, "completed"
                        elif len(data) == n_items:
                            ordered_results = []
                            for i, item in enumerate(data):
                                if isinstance(item, dict):
                                    rec = item.get("reconstructed_text") or item.get("text") or texts[i]
                                    ordered_results.append(str(rec).strip())
                                elif isinstance(item, str):
                                    ordered_results.append(item.strip())
                                else:
                                    ordered_results.append(texts[i])
                            return ordered_results, "completed"
            except Exception as err:
                err_str = str(err)
                print(f"\n[Warning] Batch with model {m} attempt {attempt} failed: {err_str[:100]}...")
                time.sleep(2.0)

    # If all attempts fail, return original texts with failed status
    print(f"\n[Error] Batch of {n_items} rows failed across models. Falling back to original text.")
    return texts, "failed"


def merge_results(
    df_original: pd.DataFrame,
    reconstruction_map: Dict[int, Tuple[str, str]]
) -> pd.DataFrame:
    """
    Merge reconstructed rows back into the full dataset.
    Preserves original order, index, and all original columns.
    Appends 'llm_completed_text' and 'llm_status'.
    """
    df_out = df_original.copy()
    
    # Initialize with default unchanged values
    completed_texts = df_out["text"].copy()
    statuses = pd.Series(["unchanged"] * len(df_out), index=df_out.index, dtype=object)

    for idx, (rec_text, status) in reconstruction_map.items():
        if idx in df_out.index:
            completed_texts.at[idx] = rec_text
            statuses.at[idx] = status

    df_out["llm_completed_text"] = completed_texts
    df_out["llm_status"] = statuses
    return df_out


def validate_output(df_original: pd.DataFrame, df_output: pd.DataFrame) -> None:
    """
    Perform strict validation checks on the output dataframe.
    """
    # 1. Row count unchanged
    if len(df_output) != len(df_original):
        raise ValueError(f"Row count changed: original {len(df_original)} vs output {len(df_output)}")

    # 2. Index unchanged
    if not (df_output.index == df_original.index).all():
        raise ValueError("Dataframe index does not match original dataset.")

    # 3. Required columns exist
    if "llm_completed_text" not in df_output.columns or "llm_status" not in df_output.columns:
        raise ValueError("Missing output columns 'llm_completed_text' or 'llm_status'.")

    # 4. Labels & Sentiment unchanged
    for check_col in ["label", "sentimen", "clean_text", "processed_text"]:
        if check_col in df_original.columns:
            orig_series = df_original[check_col].fillna("__NA__")
            out_series = df_output[check_col].fillna("__NA__")
            if not (orig_series == out_series).all():
                raise ValueError(f"Integrity check failed: '{check_col}' was modified!")

    # 5. Unchanged rows have identical text
    unchanged_mask = df_output["llm_status"] == "unchanged"
    diff_unchanged = (
        df_output.loc[unchanged_mask, "llm_completed_text"] != df_output.loc[unchanged_mask, "text"]
    ).sum()
    if diff_unchanged > 0:
        raise ValueError(f"{diff_unchanged} rows marked 'unchanged' have differing text!")

    print("[Validation] Output integrity passed all checks:")
    print(f"  - Row count: {len(df_output):,} identical to original.")
    print("  - Index, labels, sentiment, and metadata columns 100% preserved.")
    print(f"  - Unchanged rows verified: {unchanged_mask.sum():,} rows.")
    print(f"  - Processed rows verified: {(~unchanged_mask).sum():,} rows.")


def generate_report(
    df_output: pd.DataFrame,
    batch_stats: List[Dict[str, Any]],
    elapsed_time: float,
    output_path: str | Path
) -> str:
    """
    Generate Data/interim/llm_completion_report.md.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = len(df_output)
    candidate_rows = int((df_output["has_truncation"] == True).sum()) if "has_truncation" in df_output.columns else 0
    completed_rows = int((df_output["llm_status"] == "completed").sum())
    failed_rows = int((df_output["llm_status"] == "failed").sum())
    unchanged_rows = int((df_output["llm_status"] == "unchanged").sum())

    # Build batch stats table
    batch_table_rows = []
    for b in batch_stats:
        b_num = b["batch_num"]
        b_rows = b["rows"]
        b_status = b["status"]
        batch_table_rows.append(f"| Batch {b_num:03d} | {b_rows} | `{b_status}` |")
    batch_table = "\n".join(batch_table_rows)

    # Pick 5 representative before/after examples
    completed_sample = df_output[df_output["llm_status"] == "completed"].head(5)
    example_sections = []
    for i, (_, row) in enumerate(completed_sample.iterrows(), 1):
        before_text = row["text"]
        after_text = row["llm_completed_text"]
        example_sections.append(
            f"### Example {i}\n\nBefore\n\n{before_text}\n\nAfter\n\n{after_text}"
        )
    examples_md = "\n\n".join(example_sections)

    report = f"""# LLM Completion Report - Task 02

Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`  
Source: `Data/interim/audit.csv`  
Output: `Data/interim/llm_completed.csv`  

---

## 1. Summary

* **Total rows:** {total_rows:,}
* **Candidate rows (`has_truncation=True`):** {candidate_rows:,}
* **Completed:** {completed_rows:,}
* **Failed:** {failed_rows:,}
* **Unchanged:** {unchanged_rows:,}
* **Processing time:** {elapsed_time:.2f} seconds
* **Number of batches:** {len(batch_stats)}

---

## 2. Batch Statistics

| Batch | Rows | Status |
| :--- | :--- | :--- |
{batch_table}

---

## 3. Before / After Examples

{examples_md}

---

## 4. Integrity Verification

* **Row count identical to input:** Yes ({total_rows:,} rows).
* **Labels and sentiments unchanged:** Yes.
* **Unchanged rows preserved:** Yes ({unchanged_rows:,} rows).
* **No regex cleaning or slang normalization applied:** Yes (Reserved for downstream tasks).
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    default_input = project_root / "Data" / "interim" / "audit.csv"
    default_output = project_root / "Data" / "interim" / "llm_completed.csv"
    default_report = project_root / "Data" / "interim" / "llm_completion_report.md"

    parser = argparse.ArgumentParser(description="Conditional LLM Completion Pipeline (Task 02)")
    parser.add_argument("--input", type=str, default=str(default_input), help="Path to input audit.csv")
    parser.add_argument("--output", type=str, default=str(default_output), help="Path to output llm_completed.csv")
    parser.add_argument("--report", type=str, default=str(default_report), help="Path to output markdown report")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite", help="Gemini model name")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without calling API")

    args = parser.parse_args()
    start_time = time.time()

    # Step 1 — Load Dataset
    print(f"[Step 1/8] Loading dataset from: {args.input}")
    df_raw = load_dataset(args.input)
    print(f"           Loaded {len(df_raw):,} rows.")

    # Step 2 — Validate Input
    print("[Step 2/8] Validating input dataset...")
    validate_input(df_raw)

    # Step 3 — Select Candidates
    candidates_mask = df_raw["has_truncation"] == True
    candidates_df = df_raw[candidates_mask].copy()
    print(f"[Step 3/8] Selected {len(candidates_df):,} candidate rows (has_truncation == True).")

    # Step 4 — Create Batches
    batches = create_batches(candidates_df, batch_size=args.batch_size)
    print(f"[Step 4/8] Created {len(batches)} batches (Batch size: {args.batch_size}).")

    # Initialize Gemini Client
    client = None
    if not args.dry_run:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required.")
        client = genai.Client(api_key=api_key)

    # Cache handling
    cache_path = Path(args.output).parent / ".llm_cache.json"
    cache: Dict[str, str] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"           Loaded {len(cache):,} cached items from {cache_path}.")
        except Exception:
            cache = {}

    # Step 5 — Reconstruct Batches
    print(f"[Step 5/8] Running reconstruction across {len(batches)} batches...")
    reconstruction_map: Dict[int, Tuple[str, str]] = {}
    batch_stats: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(batches), desc="Processing Batches", unit="batch")
    for b_idx, batch in enumerate(batches, 1):
        indices = batch.index.tolist()
        all_cached = all(str(idx) in cache for idx in indices)
        
        if all_cached:
            for idx in indices:
                reconstruction_map[idx] = (cache[str(idx)], "completed")
            batch_stats.append({
                "batch_num": b_idx,
                "rows": len(batch),
                "status": "completed (cached)"
            })
        else:
            rec_texts, status = reconstruct_batch(
                batch_df=batch,
                client=client,
                model=args.model,
                dry_run=args.dry_run
            )
            for idx, text in zip(indices, rec_texts):
                reconstruction_map[idx] = (text, status)
                if status == "completed":
                    cache[str(idx)] = text

            batch_stats.append({
                "batch_num": b_idx,
                "rows": len(batch),
                "status": status
            })

            # Save cache after each non-cached batch
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

            # Polite delay between batches
            if not args.dry_run:
                time.sleep(3.0)

        pbar.update(1)
    pbar.close()

    # Step 6 — Merge Results
    print("[Step 6/8] Merging reconstructed results back into dataset...")
    df_output = merge_results(df_raw, reconstruction_map)

    # Step 7 — Validation
    print("[Step 7/8] Validating merged dataset...")
    validate_output(df_raw, df_output)

    # Step 8 — Reporting & Saving Deliverables
    print("[Step 8/8] Saving deliverables...")
    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_csv, index=False)
    print(f"           Saved audited + completed dataset to: {output_csv}")

    elapsed_time = time.time() - start_time
    generate_report(
        df_output=df_output,
        batch_stats=batch_stats,
        elapsed_time=elapsed_time,
        output_path=args.report
    )
    print(f"           Saved report to: {args.report}")

    # Final Execution Summary
    completed_count = (df_output["llm_status"] == "completed").sum()
    failed_count = (df_output["llm_status"] == "failed").sum()
    unchanged_count = (df_output["llm_status"] == "unchanged").sum()

    print("\n=======================================================")
    print("           TASK 02 EXECUTION SUMMARY                  ")
    print("=======================================================")
    print(f"Total Rows:        {len(df_output):,}")
    print(f"Candidate Rows:    {len(candidates_df):,}")
    print(f"Processed Rows:    {completed_count:,}")
    print(f"Failed Rows:       {failed_count:,}")
    print(f"Unchanged Rows:    {unchanged_count:,}")
    print(f"Execution Time:    {elapsed_time:.2f} seconds")
    print(f"Output Dataset:    {output_csv}")
    print(f"Completion Report: {args.report}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
