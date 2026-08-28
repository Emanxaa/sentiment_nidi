"""Phase 3 — LLM Re-Annotation (PRD).

20 batch x 50 tweet memakai GPT-4o-mini (default). Input: text_with_emoticon.
- --dry-run  : annotator deterministik (tanpa API key) untuk menguji alur penuh.
- Resume     : batch yang sudah ada di Annotation/results/ dilewati.

Output per batch:
- Annotation/results/batch_001.csv ... batch_020.csv
- Annotation/prompts/batch_001.txt ... (log prompt)
- Annotation/batches/batch_001.json ... (respons mentah)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quality_pipeline import config as C
from quality_pipeline.prompts import build_batch_prompt, validate_label
from quality_pipeline.utils import load_csv, write_csv

OUTPUT_COLUMNS = [
    "id", "text_with_emoticon", "sentimen_old", "label_old",
    "label_llm", "confidence", "reason",
]


def _rows_to_records(gold: pd.DataFrame) -> list[dict]:
    gold = gold.sort_values("id")
    return gold.to_dict("records")


def make_batches(rows: list[dict], batch_size: int = C.BATCH_SIZE) -> list[list[dict]]:
    return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]


# ---------------------------------------------------------------------------
# Annotator dry-run (deterministik, tanpa biaya API)
# ---------------------------------------------------------------------------
def annotate_batch_dry(rows: list[dict], batch_index: int) -> list[dict]:
    rng = np.random.default_rng(42 + batch_index)
    labels = ("negatif", "netral", "positif")
    out = []
    for r in rows:
        old = r["sentimen"]
        if rng.random() < 0.8:  # 80% dipertahankan
            label = old
        else:  # 20% dibalik (untuk menguji alur QA/Phase 4-5)
            label = rng.choice([l for l in labels if l != old])
        confidence = int(rng.integers(55, 100))
        out.append({
            "id": int(r["id"]),
            "label": label,
            "confidence": confidence,
            "reason": f"(dry-run) label simulasi untuk id {r['id']}",
        })
    return out


# ---------------------------------------------------------------------------
# Annotator nyata (OpenAI)
# ---------------------------------------------------------------------------
def _openai_client():
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Package 'openai' belum terinstall. Jalankan: pip install openai"
        ) from e
    return OpenAI()


def _call_batch(rows: list[dict], model: str, client) -> list[dict]:
    system, user = build_batch_prompt(rows)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    data = json.loads(content)
    anns = data.get("annotations")
    if not isinstance(anns, list):
        raise ValueError("Respons LLM tidak mengandung kunci 'annotations' array.")
    return anns


def annotate_batch_real(rows: list[dict], model: str, client, max_retries: int = 3) -> list[dict]:
    """Panggil LLM dgn retry; jika gagal, pecah menjadi chunk 10."""
    for attempt in range(1, max_retries + 1):
        try:
            return _call_batch(rows, model, client)
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries:
                print(f"    batch gagal {attempt}x; pecah jadi chunk 10. Error: {e}")
                merged = []
                for chunk in make_batches(rows, 10):
                    merged.extend(_call_batch(chunk, model, client))
                return merged
            wait = 2 ** attempt
            print(f"    retry {attempt}/{max_retries} dalam {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _coerce_conf(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _merge_annotations(rows: list[dict], anns: list[dict]) -> list[dict]:
    ann_by_id = {}
    for a in anns:
        try:
            ann_by_id[int(a["id"])] = a
        except (KeyError, TypeError, ValueError):  # noqa: PERF203
            raise ValueError(f"Anotasi tanpa id valid: {a}") from None

    records = []
    for r in rows:
        a = ann_by_id.get(int(r["id"]))
        if a is None:
            raise ValueError(f"id {r['id']} tidak ada dalam respons LLM.")
        label = str(a.get("label", "")).strip().lower()
        if not validate_label(label):
            raise ValueError(f"Label tidak valid '{label}' untuk id {r['id']}.")
        records.append({
            "id": int(r["id"]),
            "text_with_emoticon": r["text_with_emoticon"],
            "sentimen_old": r["sentimen"],
            "label_old": r["label"],
            "label_llm": label,
            "confidence": _coerce_conf(a.get("confidence")),
            "reason": str(a.get("reason", "")),
        })
    return records


def run(dry_run: bool = False, model: str = C.LLM_MODEL, only_batch: int | None = None,
        force: bool = False) -> None:
    print("Phase 3 — LLM Re-Annotation")
    gold_path = C.ANNOTATION_DIR / "gold_dataset_1000.csv"
    if not gold_path.exists():
        raise SystemExit("gold_dataset_1000.csv belum ada. Jalankan Phase 2 dulu.")
    gold = load_csv(gold_path)
    rows = _rows_to_records(gold)
    batches = make_batches(rows)

    client = None if dry_run else _openai_client()
    if dry_run:
        print("  Mode DRY-RUN: annotator simulasi (tanpa API).")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    C.BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    for bi, batch in enumerate(batches, start=1):
        if only_batch is not None and bi != only_batch:
            continue
        out_csv = C.RESULTS_DIR / f"batch_{bi:03d}.csv"
        if out_csv.exists() and not force:
            print(f"  batch {bi:02d}: sudah ada, dilewati (resume).")
            done += 1
            continue

        print(f"  batch {bi:02d}/20 ({len(batch)} tweet)...", end=" ", flush=True)
        if dry_run:
            anns = annotate_batch_dry(batch, bi)
        else:
            anns = annotate_batch_real(batch, model, client)
        records = _merge_annotations(batch, anns)
        write_csv(pd.DataFrame(records, columns=OUTPUT_COLUMNS), out_csv)

        system, user = build_batch_prompt(batch)
        (C.PROMPTS_DIR / f"batch_{bi:03d}.txt").write_text(system + "\n\n" + user, encoding="utf-8")
        (C.BATCHES_DIR / f"batch_{bi:03d}.json").write_text(
            json.dumps(anns, ensure_ascii=False, indent=2), encoding="utf-8")
        print("OK")
        done += 1

    print(f"  Selesai: {done} batch diproses -> Annotation/results/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 — LLM Re-Annotation (OpenAI GPT-4o-mini)")
    ap.add_argument("--dry-run", action="store_true", help="Annotator simulasi tanpa API")
    ap.add_argument("--model", default=C.LLM_MODEL)
    ap.add_argument("--only-batch", type=int, default=None, help="Proses satu batch saja")
    ap.add_argument("--force", action="store_true", help="Timpa batch yang sudah ada")
    args = ap.parse_args()
    run(dry_run=args.dry_run, model=args.model, only_batch=args.only_batch, force=args.force)


if __name__ == "__main__":
    main()
