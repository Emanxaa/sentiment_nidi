"""P0 — Verifikasi metrik evaluasi model (akurasi/F1/confusion matrix).

Mendeteksi metrik yang menyesatkan SEBELUM dipakai untuk tuning atau tesis:
- model collapse (selalu menebak kelas mayoritas -> Macro F1 ~0.239 utk 3 kelas),
- kelas yang tidak pernah diprediksi,
- accuracy/F1 tidak lebih baik dari baseline mayoritas,
- label di luar rentang yang diharapkan (indikasi label mapping salah).

Referensi: docs/P0_VERIFIKASI_EVALUASI.md

Cara pakai:
    python -m quality_pipeline.verify_metrics --preds <csv_prediksi> \
        --y-true <kolom_label_aktual> --y-pred <kolom_label_prediksi> \
        --label-names negatif,netral,positif

Contoh:
    python -m quality_pipeline.verify_metrics \
        --preds hasil_prediksi_test_indobertweet_lora.csv \
        --y-true label_aktual --y-pred label_prediksi \
        --label-names negatif,netral,positif
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from quality_pipeline import config as C
from quality_pipeline.utils import load_csv


def majority_baseline(y_true: np.ndarray) -> tuple[float, float]:
    """Metrik yang diperoleh jika model SELALU menebak kelas mayoritas.

    Mengembalikan (accuracy, macro_f1). Untuk 3 kelas dengan proporsi
    mayoritas p: macro_f1 = (2p/(1+p))/3. Inilah nilai ~0.2393 pada grid
    search LSTM yang collapse (p ~ 0.56).
    """
    y = np.asarray(y_true)
    vals, counts = np.unique(y, return_counts=True)
    maj = vals[int(np.argmax(counts))]
    p_maj = float(counts.max() / len(y))
    y_pred_maj = np.full_like(y, maj)
    acc = float(accuracy_score(y, y_pred_maj))
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y, y_pred_maj, average="macro", zero_division=0
    )
    return acc, float(f1_macro)


def full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """Hitung seluruh metrik inti dengan parameter eksplisit & konsisten.

    - accuracy, precision/recall/f1 macro, weighted f1
    - classification report per kelas (labels eksplisit 0..n-1)
    - confusion matrix (labels eksplisit agar tidak bergantung urutan)
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(label_names) if label_names else len(np.unique(np.concatenate([yt, yp])))
    labels = list(range(n))

    prec, rec, f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )
    _, _, f1w, _ = precision_recall_fscore_support(
        yt, yp, average="weighted", zero_division=0
    )
    report = classification_report(
        yt, yp, labels=labels, target_names=label_names, zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=labels)

    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "f1_weighted": float(f1w),
        "classification_report": report,
        "confusion_matrix": cm,
        "n_samples": int(len(yt)),
    }


def sanity_flags(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """Deteksi kondisi degenerate pada prediksi.

    Mengembalikan dict berisi distribusi, flag collapse, dan peringatan.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(label_names) if label_names else len(np.unique(np.concatenate([yt, yp])))
    m = full_metrics(yt, yp, label_names)
    acc_base, f1_base = majority_baseline(yt)

    pred_counts = pd.Series(yp).value_counts().sort_index()
    true_counts = pd.Series(yt).value_counts().sort_index()
    never_predicted = [int(c) for c in range(n) if c not in pred_counts.index]
    out_of_range = sorted(
        {int(x) for x in np.concatenate([np.unique(yt), np.unique(yp)])} - set(range(n))
    )

    warnings: list[str] = []
    if m["f1_macro"] <= f1_base + 1e-6:
        warnings.append(
            f"COLLAPSE: Macro F1 {m['f1_macro']:.4f} <= baseline mayoritas "
            f"{f1_base:.4f} - model hampir pasti selalu menebak satu kelas."
        )
    if m["accuracy"] <= acc_base + 1e-6:
        warnings.append(
            f"Accuracy {m['accuracy']:.4f} <= baseline {acc_base:.4f} - model "
            "tidak lebih baik dari tebakan mayoritas."
        )
    if never_predicted:
        warnings.append(f"Kelas tak pernah diprediksi: {never_predicted}")
    if out_of_range:
        warnings.append(
            f"Label di luar rentang 0..{n-1}: {out_of_range} - cek label mapping!"
        )

    return {
        "true_distribution": true_counts.to_dict(),
        "pred_distribution": pred_counts.to_dict(),
        "classes_never_predicted": never_predicted,
        "is_collapse": bool(m["f1_macro"] <= f1_base + 1e-6),
        "baseline_accuracy": acc_base,
        "baseline_macro_f1": f1_base,
        "warnings": warnings,
    }


def _format_cm(cm: np.ndarray, label_names: list[str] | None) -> str:
    headers = label_names or [str(i) for i in range(cm.shape[0])]
    lines = ["| | " + " | ".join(headers) + " |", "|---|" + "---|" * cm.shape[0]]
    for i, name in enumerate(headers):
        lines.append(f"| {name} | " + " | ".join(str(int(x)) for x in cm[i]) + " |")
    return "\n".join(lines)


def run(
    preds_csv: str,
    y_true_col: str,
    y_pred_col: str,
    label_names: list[str],
    out_md: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Muat CSV prediksi, verifikasi metrik, cetak laporan, simpan markdown."""
    df = load_csv(preds_csv)
    for col in (y_true_col, y_pred_col):
        if col not in df.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ada di {preds_csv}. "
                f"Kolom tersedia: {df.columns.tolist()}"
            )

    y_true = pd.to_numeric(df[y_true_col], errors="coerce")
    y_pred = pd.to_numeric(df[y_pred_col], errors="coerce")
    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError(
            "Kolom label harus berupa angka (0/1/2). Ada nilai non-numerik atau NaN."
        )
    y_true = y_true.astype(int).values
    y_pred = y_pred.astype(int).values
    if len(y_true) != len(y_pred):
        raise ValueError(f"Panjang y_true ({len(y_true)}) != y_pred ({len(y_pred)}).")

    m = full_metrics(y_true, y_pred, label_names)
    s = sanity_flags(y_true, y_pred, label_names)

    lines = [
        "# Verifikasi Metrik Evaluasi",
        "",
        f"- File prediksi : `{preds_csv}`",
        f"- Kolom y_true  : `{y_true_col}`",
        f"- Kolom y_pred  : `{y_pred_col}`",
        f"- Jumlah sampel : **{m['n_samples']}**",
        "",
        "## Distribusi",
        "",
        f"- Aktual   : {s['true_distribution']}",
        f"- Prediksi : {s['pred_distribution']}",
        "",
        "## Metrik",
        "",
        "| Metrik | Nilai |",
        "|---|---|",
        f"| Accuracy | {m['accuracy']:.6f} |",
        f"| Precision Macro | {m['precision_macro']:.6f} |",
        f"| Recall Macro | {m['recall_macro']:.6f} |",
        f"| Macro F1 | {m['f1_macro']:.6f} |",
        f"| Weighted F1 | {m['f1_weighted']:.6f} |",
        f"| Baseline mayoritas (acc) | {s['baseline_accuracy']:.6f} |",
        f"| Baseline mayoritas (macro F1) | {s['baseline_macro_f1']:.6f} |",
        "",
        "## Status",
        "",
        f"- **COLLAPSE** : {'YA' if s['is_collapse'] else 'TIDAK'}",
        "",
    ]
    if s["warnings"]:
        lines += ["## Peringatan", ""]
        lines += [f"- ! {w}" for w in s["warnings"]]
        lines += [""]
    lines += [
        "## Classification Report",
        "",
        "```",
        m["classification_report"].rstrip(),
        "```",
        "",
        "## Confusion Matrix (baris=aktual, kolom=prediksi)",
        "",
        _format_cm(m["confusion_matrix"], label_names),
        "",
    ]
    text = "\n".join(lines)

    # Cetak ringkas ke terminal
    print(f"=== VERIFIKASI METRIK: {preds_csv} ===")
    print(f"Sampel        : {m['n_samples']}")
    print(f"Distribusi    : aktual={s['true_distribution']} pred={s['pred_distribution']}")
    print(f"Accuracy      : {m['accuracy']:.6f}  (baseline {s['baseline_accuracy']:.6f})")
    print(f"Macro F1      : {m['f1_macro']:.6f}  (baseline {s['baseline_macro_f1']:.6f})")
    print(f"Weighted F1   : {m['f1_weighted']:.6f}")
    print(f"COLLAPSE      : {'YA' if s['is_collapse'] else 'TIDAK'}")
    if s["warnings"]:
        for w in s["warnings"]:
            print(f"  ! {w}")
    print()

    if out_md and not dry_run:
        out_path = C.REPORTS_DIR / out_md
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Laporan lengkap -> {out_path}")

    return {"metrics": m, "flags": s}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="P0 — Verifikasi metrik evaluasi model (deteksi collapse/label mapping)."
    )
    ap.add_argument("--preds", required=True, help="CSV berisi label aktual & prediksi.")
    ap.add_argument("--y-true", required=True, help="Nama kolom label aktual.")
    ap.add_argument("--y-pred", required=True, help="Nama kolom label prediksi.")
    ap.add_argument(
        "--label-names",
        default="negatif,netral,positif",
        help="Nama kelas dipisah koma (default: negatif,netral,positif).",
    )
    ap.add_argument("--out", default="verifikasi_evaluasi.md", help="Nama file laporan di reports/.")
    ap.add_argument("--dry-run", action="store_true", help="Cetak saja, tanpa menulis laporan.")
    args = ap.parse_args()

    label_names = [n.strip() for n in args.label_names.split(",") if n.strip()]
    run(
        args.preds,
        args.y_true,
        args.y_pred,
        label_names,
        out_md=args.out,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
