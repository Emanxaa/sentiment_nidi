"""Phase 4 — Quality Assurance (PRD).

- Merge 20 batch -> Annotation/reports/merged_annotations.csv
- Flag review (confidence < 80 ATAU label berubah) -> Annotation/reports/qa_flags.csv
- Template review manusia -> Annotation/human_review.csv (diisi manual)
- Bentuk Data/data_banjir_v2.csv (schema sama dgn data_banjir.csv):
    * teks dibersihkan dgn cleaning rules Task 0.4
    * label memakai hasil LLM untuk baris diterima / review_label untuk yang direview
    * baris pending (perlu review tapi belum diisi) mempertahankan label asli
"""
from __future__ import annotations

import argparse

import pandas as pd

from quality_pipeline import config as C
from quality_pipeline.utils import apply_cleaning_rules, load_csv, write_csv

ANNOTATION_COLUMNS = [
    "id", "text_with_emoticon", "sentimen_old", "label_old",
    "label_llm", "confidence", "reason",
]


def load_merged() -> pd.DataFrame:
    files = sorted(C.RESULTS_DIR.glob("batch_*.csv"))
    if not files:
        raise SystemExit("Belum ada batch di Annotation/results/. Jalankan Phase 3 dulu.")
    frames = [load_csv(f) for f in files]
    merged = pd.concat(frames, ignore_index=True)
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")
    merged["need_review"] = (
        (merged["confidence"].fillna(0) < C.CONFIDENCE_THRESHOLD)
        | (merged["label_llm"] != merged["sentimen_old"])
    )
    return merged


def load_human_review() -> dict[int, str]:
    path = C.ANNOTATION_DIR / "human_review.csv"
    if not path.exists():
        return {}
    hr = load_csv(path)
    hr = hr[hr["review_label"].notna() & (hr["review_label"].astype(str).str.strip() != "")]
    return {int(r["id"]): str(r["review_label"]).strip().lower() for _, r in hr.iterrows()}


def build_v2(merged: pd.DataFrame, review_map: dict[int, str]) -> tuple[pd.DataFrame, int, int]:
    raw = load_csv(C.RAW_CSV)
    raw["text"] = raw["text"].apply(apply_cleaning_rules)  # Task 0.4

    final_label: dict[int, str] = {}
    n_reviewed = 0
    n_pending = 0
    for _, r in merged.iterrows():
        iid = int(r["id"])
        if r["need_review"]:
            if iid in review_map:
                final_label[iid] = review_map[iid]
                n_reviewed += 1
            else:
                n_pending += 1  # pertahankan label asli
        else:
            final_label[iid] = r["label_llm"]

    def resolve(row):
        iid = int(row.name)
        lab = final_label.get(iid)
        if lab and lab in C.LABEL_MAP:
            return lab
        return row["sentimen"]

    raw["sentimen"] = raw.apply(resolve, axis=1)
    raw["label"] = raw["sentimen"].map(C.LABEL_MAP).astype(int)
    return raw, n_reviewed, n_pending


def run(dry_run: bool = False) -> None:
    print("Phase 4 — Quality Assurance")
    merged = load_merged()
    write_csv(merged, C.ANN_REPORTS_DIR / "merged_annotations.csv")
    print(f"  Merge: {len(merged)} baris -> Annotation/reports/merged_annotations.csv")

    qa = merged[merged["need_review"]].copy()
    write_csv(qa, C.ANN_REPORTS_DIR / "qa_flags.csv")
    print(f"  Flag review: {len(qa)} baris (confidence<{C.CONFIDENCE_THRESHOLD} atau label berubah)")

    review_map = load_human_review()
    if review_map:
        print(f"  Review manusia terbaca: {len(review_map)} baris terisi")
    else:
        template = qa[["id", "text_with_emoticon", "sentimen_old", "label_llm", "confidence", "reason"]].copy()
        template["review_label"] = ""
        write_csv(template, C.ANNOTATION_DIR / "human_review.csv")
        print("  Template review -> Annotation/human_review.csv (isi kolom review_label, lalu jalankan ulang)")

    v2, n_reviewed, n_pending = build_v2(merged, review_map)
    write_csv(v2, C.V2_CSV)
    print(f"  data_banjir_v2.csv: {len(v2)} baris (label diterapkan: {len(merged) - n_pending - n_reviewed}, direview: {n_reviewed}, pending: {n_pending})")

    lines = [
        "# Laporan QA (Phase 4)",
        "",
        f"- Total anotasi digabung: **{len(merged)}**",
        f"- Flag review (confidence < {C.CONFIDENCE_THRESHOLD} atau label berubah): **{len(qa)}**",
        f"- Label LLM diterapkan (tanpa review): **{len(merged) - len(qa)}**",
        f"- Direview manusia: **{n_reviewed}**",
        f"- Pending (label asli dipertahankan): **{n_pending}**",
        "",
        f"- Output: `Data/{C.V2_CSV.name}`",
        "",
    ]
    (C.ANN_REPORTS_DIR / "qa_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("  Ringkasan -> Annotation/reports/qa_report.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 — Quality Assurance")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
