"""Phase 0 — Data Quality Audit (Task 0.1–0.4 PRD).

Menghasilkan:
- reports/duplicate_report.csv
- reports/missing_report.csv
- reports/noise_pattern_report.csv
- reports/data_quality_audit.md
"""
from __future__ import annotations

import argparse

import pandas as pd

from quality_pipeline import config as C
from quality_pipeline.utils import (
    apply_cleaning_rules,
    has_math_letters,
    load_csv,
    near_duplicate_pairs,
    norm_for_dup,
    safe_str,
    write_csv,
)

MISSING_COLUMNS = ["text", "sentimen", "emoticon", "keyword", "created_at"]


def audit_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Task 0.1 — exact, near duplicate, repost."""
    text = df["text"].astype(str).str.strip()
    rows: list[dict] = []

    # --- Exact duplicate & repost ---
    dup_mask = text.duplicated(keep=False)
    if dup_mask.any():
        grouped = text[dup_mask].groupby(text[dup_mask])
        for tval, grp in grouped:
            ids = grp.index.tolist()
            if len(ids) < 2:
                continue
            dates = [safe_str(d) for d in df.loc[ids, "created_at"]]
            n_dates = len({d for d in dates if d})
            kategori = "repost" if n_dates > 1 else "exact_duplicate"
            alasan = (
                f"Teks identik; {len(ids)} baris, {n_dates} tanggal berbeda"
                if n_dates > 1
                else f"Teks identik; {len(ids)} baris"
            )
            for pos, i in enumerate(ids):
                rows.append({
                    "id": i,
                    "kategori": kategori,
                    "status": "asli" if pos == 0 else "duplikat",
                    "text": tval[:300],
                    "created_at": dates[pos],
                    "pasangan_id": "|".join(str(x) for x in ids),
                    "rasio_kemiripan": 1.0,
                    "alasan": alasan,
                })

    # --- Near duplicate ---
    norms = [norm_for_dup(t) for t in df["text"]]
    for i, j, ratio in near_duplicate_pairs(norms):
        rows.append({
            "id": j,
            "kategori": "near_duplicate",
            "status": "duplikat",
            "text": safe_str(df["text"].iloc[j])[:300],
            "created_at": safe_str(df["created_at"].iloc[j]),
            "pasangan_id": str(i),
            "rasio_kemiripan": ratio,
            "alasan": f"Kemiripan {ratio:.0%} dengan id {i}",
        })

    report = pd.DataFrame(rows, columns=[
        "id", "kategori", "status", "text", "created_at", "pasangan_id",
        "rasio_kemiripan", "alasan",
    ])
    return report


def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Task 0.2 — missing value per kolom."""
    rows: list[dict] = []
    for col in MISSING_COLUMNS:
        total = len(df)
        if col not in df.columns:
            rows.append({
                "kolom": col, "total": total, "nan": total, "kosong": 0,
                "missing": total, "terisi": 0, "persentase_missing": 100.0,
                "catatan": "kolom tidak ada",
            })
            continue
        nan = int(df[col].isna().sum())
        str_col = df[col].astype(str).str.strip()
        kosong = int((str_col == "").sum())
        missing = int((df[col].isna() | (str_col == "")).sum())
        rows.append({
            "kolom": col, "total": total, "nan": nan, "kosong": kosong,
            "missing": missing, "terisi": total - missing,
            "persentase_missing": round(100 * missing / total, 2),
            "catatan": "",
        })
    return pd.DataFrame(rows)


def audit_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Task 0.3 — deteksi pola noise + artefak scraping."""
    text = df["text"].astype(str)
    rows: list[dict] = []
    for name, pattern in C.NOISE_PATTERNS.items():
        mask = text.str.contains(pattern, na=False)
        n = int(mask.sum())
        contoh = text.index[mask][:5].tolist()
        rows.append({
            "pola": name,
            "deskripsi": pattern.pattern,
            "jumlah_baris": n,
            "persentase": round(100 * n / len(df), 2),
            "contoh_id": "|".join(str(x) for x in contoh),
        })
    # artefak huruf matematika (bukan dari NOISE_PATTERNS karena bukan regex sederhana)
    math_mask = text.map(has_math_letters)
    n_math = int(math_mask.sum())
    rows.append({
        "pola": "artefak_huruf_matematika",
        "deskripsi": "Karakter Mathematical Alphanumeric U+1D400-U+1D7FF (artefak scraping)",
        "jumlah_baris": n_math,
        "persentase": round(100 * n_math / len(df), 2),
        "contoh_id": "|".join(str(x) for x in text.index[math_mask][:5].tolist()),
    })
    return pd.DataFrame(rows)


def _clean_preview(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Contoh sebelum/sesudah Task 0.4 cleaning rules (untuk laporan md)."""
    rows = []
    for i in df.sample(n, random_state=C.RANDOM_STATE).index:
        before = safe_str(df.loc[i, "text"])
        after = apply_cleaning_rules(before)
        if before != after:
            rows.append({
                "id": i, "sebelum": before[:250], "sesudah": after[:250],
            })
    return pd.DataFrame(rows)


def run(dry_run: bool = False) -> None:
    print("Phase 0 — Data Quality Audit")
    df = load_csv(C.RAW_CSV)
    print(f"  Memuat {len(df)} baris dari {C.RAW_CSV.name}")

    dup = audit_duplicates(df)
    write_csv(dup, C.REPORTS_DIR / "duplicate_report.csv")
    n_exact = int(((dup["kategori"] == "exact_duplicate") & (dup["status"] == "duplikat")).sum())
    n_repost = int(((dup["kategori"] == "repost") & (dup["status"] == "duplikat")).sum())
    n_near = int((dup["kategori"] == "near_duplicate").sum())
    print(f"  Duplicate: exact={n_exact}, repost={n_repost}, near={n_near}")

    miss = audit_missing(df)
    write_csv(miss, C.REPORTS_DIR / "missing_report.csv")
    print(f"  Missing: {len(miss)} kolom diaudit")

    noise = audit_noise(df)
    write_csv(noise, C.REPORTS_DIR / "noise_pattern_report.csv")
    print(f"  Noise: {len(noise)} pola terdeteksi")

    preview = _clean_preview(df)
    preview.to_csv(C.REPORTS_DIR / "cleaning_rules_preview.csv", index=False, encoding="utf-8")

    # ---- Ringkasan markdown ----
    pct_dup = round(100 * (n_exact + n_repost) / len(df), 2)
    lines = [
        "# Laporan Data Quality Audit",
        "",
        f"**Sumber:** `Data/{C.RAW_CSV.name}` · **Jumlah baris:** {len(df)}",
        "",
        "## Ringkasan vs Success Metrics",
        "",
        "| Metrik | Hasil | Target |",
        "|---|---|---|",
        f"| Missing value | {miss['persentase_missing'].max():.2f}% (tertinggi: {miss.loc[miss['persentase_missing'].idxmax(), 'kolom']}) | <1% |",
        f"| Duplicate (exact+repost) | {pct_dup:.2f}% ({n_exact + n_repost} baris) | <5% |",
        f"| Near duplicate | {n_near} pasangan | — |",
        f"| Noise (baris kena pola) | {noise['jumlah_baris'].sum()} kemunculan pola | <3% |",
        "",
        "## Task 0.1 — Duplicate Audit",
        "",
        f"- Exact duplicate: **{n_exact}** baris",
        f"- Repost (teks sama, tanggal beda): **{n_repost}** baris",
        f"- Near duplicate: **{n_near}** baris",
        "- Detail: `reports/duplicate_report.csv`",
        "",
        "## Task 0.2 — Missing Value Audit",
        "",
        "| Kolom | Missing | Persentase |",
        "|---|---|---|",
    ]
    for _, r in miss.iterrows():
        lines.append(f"| {r['kolom']} | {r['missing']} | {r['persentase_missing']:.2f}% |")
    lines += [
        "",
        "## Task 0.3 — Noise Detection",
        "",
        "| Pola | Jumlah baris | Persentase |",
        "|---|---|---|",
    ]
    for _, r in noise.iterrows():
        lines.append(f"| {r['pola']} | {r['jumlah_baris']} | {r['persentase']:.2f}% |")
    lines += [
        "",
        "## Task 0.4 — Cleaning Rules (contoh)",
        "",
        "Contoh sebelum/sesudah penerapan aturan baru (lengkap: `reports/cleaning_rules_preview.csv`):",
        "",
    ]
    if len(preview):
        for _, r in preview.head(5).iterrows():
            lines.append(f"- **id {r['id']}**\n  - sebelum: `{r['sebelum']}`\n  - sesudah: `{r['sesudah']}`")
    else:
        lines.append("- Tidak ada contoh berubah pada sampel.")
    lines.append("")
    (C.REPORTS_DIR / "data_quality_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("  Laporan -> reports/data_quality_audit.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 — Data Quality Audit")
    ap.add_argument("--dry-run", action="store_true", help="Tidak dipakai di Phase 0 (read-only audit)")
    args = ap.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
