"""Phase 5 — Label Quality Evaluation (PRD).

Menghitung:
- Agreement rate (label asli vs LLM)
- Cohen's Kappa (sklearn) + interpretasi
- Label flip analysis (arah + kategori heuristik: Sarkasme, Mixed, Informational, Apresiasi, Kritik)

Output:
- Annotation/reports/evaluation_report.md
- Annotation/reports/label_flip_analysis.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from quality_pipeline import config as C
from quality_pipeline.phase4_qa import load_merged
from quality_pipeline.utils import write_csv

KAPPA_TABLE = [
    ("<0.40", "Buruk"),
    ("0.40-0.60", "Sedang"),
    ("0.60-0.80", "Baik"),
    (">0.80", "Sangat baik"),
]

# Heuristik kategori label flip (disempurnakan manual di kolom kategori_manual)
SARCASM = ["wkwk", "haha", "lol", "parah", "bego", "goblok", "bodoh", "sih", "banget", "sarkas"]
MIXED = ["tapi", "tp ", "sayangnya", "namun", "di sisi lain", "walaupun", "meski"]
INFORMATIONAL = ["informasi", "bnpb", "data", "laporan", "update", "peringatan", "imbau", "status", "susulan", "info"]
APRESIASI = ["terima kasih", "apresiasi", "salut", "semangat", "bagus", "hebat", "terimakasih", "berterima kasih", "alhamdulillah"]
KRITIK = ["buruk", "jelek", "gagal", "lambat", "protes", "korup", "goblok", "tidak becus", "asal", "diam", "parah"]


def categorize(row: pd.Series) -> str:
    t = str(row["text_with_emoticon"]).lower()
    if any(k in t for k in SARCASM):
        return "Sarkasme"
    if any(k in t for k in APRESIASI):
        return "Apresiasi"
    if any(k in t for k in KRITIK):
        return "Kritik"
    if any(k in t for k in INFORMATIONAL):
        return "Informational"
    if any(k in t for k in MIXED):
        return "Mixed sentiment"
    return "Lainnya"


def run(dry_run: bool = False) -> None:
    print("Phase 5 — Label Quality Evaluation")
    merged = load_merged()
    old = merged["sentimen_old"]
    new = merged["label_llm"]

    labels = ["negatif", "netral", "positif"]
    agreement = float((old == new).mean())
    kappa = float(cohen_kappa_score(old, new, labels=labels))
    conf = confusion_matrix(old, new, labels=labels)

    print(f"  Agreement rate : {agreement:.2%} ({int((old == new).sum())}/{len(merged)})")
    print(f"  Cohen's Kappa  : {kappa:.4f}")

    flips = merged[old != new].copy()
    flips["arah"] = flips["sentimen_old"] + " -> " + flips["label_llm"]
    flips["kategori"] = flips.apply(categorize, axis=1)
    flips["kategori_manual"] = ""  # untuk disempurnakan manual
    write_csv(flips, C.ANN_REPORTS_DIR / "label_flip_analysis.csv")
    print(f"  Label flip: {len(flips)} baris -> Annotation/reports/label_flip_analysis.csv")

    # interpretasi kappa
    interp = "Sangat baik" if kappa > 0.8 else "Baik" if kappa > 0.6 else "Sedang" if kappa >= 0.4 else "Buruk"

    lines = [
        "# Laporan Evaluasi Kualitas Label (Phase 5)",
        "",
        f"- Jumlah sampel: **{len(merged)}**",
        f"- Agreement rate: **{agreement:.2%}** ({int((old == new).sum())}/{len(merged)})",
        f"- Cohen's Kappa: **{kappa:.4f}** — interpretasi: **{interp}**",
        "",
        "## Confusion Matrix (baris=label asli, kolom=label LLM)",
        "",
        "| | negatif | netral | positif |",
        "|---|---|---|---|",
    ]
    for i, row_label in enumerate(labels):
        cells = " | ".join(str(int(x)) for x in conf[i])
        lines.append(f"| {row_label} | {cells} |")
    lines += [
        "",
        "## Interpretasi Kappa",
        "",
        "| Kappa | Interpretasi |",
        "|---|---|",
    ]
    for rng, label in KAPPA_TABLE:
        lines.append(f"| {rng} | {label} |")
    lines += [
        "",
        "## Label Flip Analysis",
        "",
        f"- Total label berubah: **{len(flips)}**",
        "- Detail per baris: `Annotation/reports/label_flip_analysis.csv`",
        "",
        "### Sebaran arah perubahan",
        "",
    ]
    dir_counts = flips["arah"].value_counts()
    for arah, cnt in dir_counts.items():
        lines.append(f"- {arah}: {cnt}")
    lines += ["", "### Sebaran kategori (heuristik, siap disempurnakan manual)", ""]
    for kat, cnt in flips["kategori"].value_counts().items():
        lines.append(f"- {kat}: {cnt}")
    lines.append("")
    (C.ANN_REPORTS_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("  Ringkasan -> Annotation/reports/evaluation_report.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 — Label Quality Evaluation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
