"""E3 — Kalibrasi threshold per-kelas dari probabilitas tersimpan (tanpa GPU).

Metode (ditulis eksplisit agar dapat direplikasi & dikoreksi):
1. Skor keputusan per kelas:  s_i = p_i * w_i   (w >= 0)
   - w = (1,1,1) ekuivalen dengan argmax softmax (baseline).
   - Menaikkan w netral membuat model "lebih berani" memilih netral.
2. Bobot w dicari HANYA di validation set (test tetap tak tersentuh) dengan
   coordinate-ascent greedy: untuk tiap kelas, coba grid w dan pertahankan
   yang memaksimalkan val macro F1; ulangi beberapa pass sampai konvergen.
3. Bobot terbaik diterapkan ke test; laporkan metrik sebelum vs sesudah,
   uji McNemar exact, dan ECE (10 bin) untuk probabilitas argmax baseline.

Catatan: threshold mengubah KEPUTUSAN, bukan keyakinan probabilitas - jadi ECE
dilaporkan untuk probabilitas baseline (tsp. penilaian kalibrasi model).

Cara pakai:
    python -m quality_pipeline.calibrate_thresholds ^
        --val probs_val.csv --test probs_test.csv ^
        --y-true label_aktual ^
        --prob-cols prob_negatif,prob_netral,prob_positif ^
        --label-names negatif,netral,positif
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from quality_pipeline import config as C
from quality_pipeline.utils import load_csv
from quality_pipeline.verify_metrics import mcnemar_exact


def load_probs(csv: str, y_true_col: str, prob_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Muat CSV probabilitas -> (y_true int, P matrix dinormalisasi sum=1)."""
    df = load_csv(csv)
    if y_true_col not in df.columns:
        raise ValueError(f"Kolom '{y_true_col}' tidak ada di {csv}. Tersedia: {df.columns.tolist()}")
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom probabilitas {missing} tidak ada di {csv}. Tersedia: {df.columns.tolist()}")
    y = df[y_true_col].astype(int).values
    P = df[prob_cols].astype(float).values
    P = P / P.sum(axis=1, keepdims=True)
    return y, P


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width binning pada confidence)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        e += (mask.sum() / total) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(e)


def predict_weighted(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Keputusan: argmax_i (p_i * w_i)."""
    return np.argmax(P * w, axis=1)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> float:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), average="macro", zero_division=0
    )
    return float(f1)


def coordinate_ascent(
    P_val: np.ndarray,
    y_val: np.ndarray,
    passes: int = 3,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Cari bobot per-kelas dengan coordinate-ascent greedy (di validation)."""
    if grid is None:
        grid = np.round(np.arange(0.50, 2.01, 0.05), 2)
    n_classes = P_val.shape[1]
    w = np.ones(n_classes)
    best = macro_f1(y_val, predict_weighted(P_val, w))
    for p in range(passes):
        improved = False
        for k in range(n_classes):
            for g in grid:
                trial = w.copy()
                trial[k] = float(g)
                f1 = macro_f1(y_val, predict_weighted(P_val, trial))
                if f1 > best + 1e-9:
                    best, w, improved = f1, trial, True
        print(f"  pass {p+1}: val macro F1 = {best:.4f}, w = {np.round(w, 3).tolist()}")
        if not improved:
            break
    return w


def run(
    val_csv: str,
    test_csv: str,
    y_true_col: str,
    prob_cols: list[str],
    label_names: list[str],
    out_md: str | None = None,
    dry_run: bool = False,
) -> dict:
    n = len(label_names)
    y_val, P_val = load_probs(val_csv, y_true_col, prob_cols)
    y_test, P_test = load_probs(test_csv, y_true_col, prob_cols)
    if P_val.shape[1] != n or P_test.shape[1] != n:
        raise ValueError(f"Jumlah kolom probabilitas ({P_val.shape[1]}) != label_names ({n}).")

    # Baseline (argmax)
    yv0 = np.argmax(P_val, axis=1)
    yt0 = np.argmax(P_test, axis=1)
    f1v0, f1t0 = macro_f1(y_val, yv0, n), macro_f1(y_test, yt0, n)
    acc_t0 = float(accuracy_score(y_test, yt0))

    # Kalibrasi di validation
    print("=== E3 KALIBRASI THRESHOLD ===")
    print(f"Val baseline  : macro F1 {f1v0:.4f}")
    w = coordinate_ascent(P_val, y_val)
    f1v1 = macro_f1(y_val, predict_weighted(P_val, w))

    # Terapkan ke test
    yt1 = predict_weighted(P_test, w)
    f1t1 = macro_f1(y_test, yt1, n)
    acc_t1 = float(accuracy_score(y_test, yt1))
    mc = mcnemar_exact(y_test, yt0, yt1)

    # ECE pada confidence argmax baseline (test)
    conf = P_test.max(axis=1)
    ece0 = ece(conf, (yt0 == y_test).astype(float))

    per_before = precision_recall_fscore_support(
        y_test, yt0, labels=list(range(n)), average=None, zero_division=0
    )
    per_after = precision_recall_fscore_support(
        y_test, yt1, labels=list(range(n)), average=None, zero_division=0
    )

    print("\n--- Test sebelum vs sesudah kalibrasi ---")
    print(f"Accuracy : {acc_t0:.4f} -> {acc_t1:.4f}")
    print(f"Macro F1 : {f1t0:.4f} -> {f1t1:.4f}")
    for i, name in enumerate(label_names):
        print(
            f"  {name:<8}: P {per_before[0][i]:.3f}->{per_after[0][i]:.3f}  "
            f"R {per_before[1][i]:.3f}->{per_after[1][i]:.3f}  "
            f"F1 {per_before[2][i]:.3f}->{per_after[2][i]:.3f}"
        )
    print(f"McNemar  : b={mc['b_a_right_b_wrong']}, c={mc['c_a_wrong_b_right']}, "
          f"p={mc['p_value']:.6f} ({'signifikan' if mc['significant_0.05'] else 'tidak signifikan'})")
    print(f"ECE (argmax baseline, test): {ece0:.4f}")
    print(f"Bobot terbaik (dari validation): {np.round(w, 3).tolist()}")

    if out_md and not dry_run:
        lines = [
            "# E3 — Hasil Kalibrasi Threshold",
            "",
            f"- Val CSV: `{val_csv}` | Test CSV: `{test_csv}`",
            f"- Bobot terbaik (coordinate-ascent di validation): **{np.round(w, 3).tolist()}**",
            f"- Val Macro F1: {f1v0:.4f} -> **{f1v1:.4f}**",
            f"- Test Macro F1: {f1t0:.4f} -> **{f1t1:.4f}** | Accuracy: {acc_t0:.4f} -> {acc_t1:.4f}",
            f"- McNemar vs argmax: b={mc['b_a_right_b_wrong']}, c={mc['c_a_wrong_b_right']}, "
            f"p={mc['p_value']:.6f} ({'signifikan' if mc['significant_0.05'] else 'tidak signifikan'})",
            f"- ECE argmax baseline (test): {ece0:.4f}",
            "",
            "| Kelas | P sebelum | P sesudah | R sebelum | R sesudah | F1 sebelum | F1 sesudah |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, name in enumerate(label_names):
            lines.append(
                f"| {name} | {per_before[0][i]:.3f} | {per_after[0][i]:.3f} | "
                f"{per_before[1][i]:.3f} | {per_after[1][i]:.3f} | "
                f"{per_before[2][i]:.3f} | {per_after[2][i]:.3f} |"
            )
        out_path = C.REPORTS_DIR / out_md
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Laporan -> {out_path}")

    return {
        "weights": w.tolist(),
        "val_f1_before": f1v0,
        "val_f1_after": f1v1,
        "test_f1_before": f1t0,
        "test_f1_after": f1t1,
        "test_acc_before": acc_t0,
        "test_acc_after": acc_t1,
        "mcnemar": mc,
        "ece": ece0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="E3 — Kalibrasi threshold per-kelas dari probabilitas.")
    ap.add_argument("--val", required=True, help="CSV probabilitas validation.")
    ap.add_argument("--test", required=True, help="CSV probabilitas test.")
    ap.add_argument("--y-true", required=True, help="Nama kolom label aktual.")
    ap.add_argument(
        "--prob-cols",
        default="prob_negatif,prob_netral,prob_positif",
        help="Nama kolom probabilitas per kelas, dipisah koma.",
    )
    ap.add_argument(
        "--label-names", default="negatif,netral,positif", help="Nama kelas dipisah koma."
    )
    ap.add_argument("--out", default="hasil_kalibrasi_threshold.md", help="Nama laporan di reports/.")
    ap.add_argument("--dry-run", action="store_true", help="Cetak saja, tanpa menulis laporan.")
    args = ap.parse_args()

    run(
        args.val,
        args.test,
        args.y_true,
        [c.strip() for c in args.prob_cols.split(",")],
        [n.strip() for n in args.label_names.split(",") if n.strip()],
        out_md=args.out,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
