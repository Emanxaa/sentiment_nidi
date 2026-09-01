"""Auto Experiment Summary: menulis metadata run (config, commit, dataset MD5) ke file & stdout."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def git_commit_short() -> str:
    """Hash commit git saat ini (atau 'unknown' bila bukan repo git)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def file_md5(path: str | Path) -> str:
    """MD5 file (untuk audit data lineage)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def experiment_summary(
    exp_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    csv_path: str | None = None,
    dataset_path: str | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Buat dict ringkasan + tulis file JSON (opsional) + cetak ke stdout.

    Metrics contoh: {"accuracy": ..., "f1_macro": ..., "netral_f1": ...}
    """
    summary = {
        "experiment": exp_id,
        "commit": git_commit_short(),
        "config": config,
        "metrics": metrics,
    }
    if csv_path:
        summary["prediction_csv"] = csv_path
    if dataset_path:
        summary["dataset_md5"] = file_md5(dataset_path)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print("AUTO EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Experiment : {exp_id}")
    print(f"Commit     : {summary.get('commit')}")
    if dataset_path:
        print(f"Dataset MD5: {summary['dataset_md5']}")
    print(f"Config     : {json.dumps(config, ensure_ascii=False)}")
    print(f"Metrics    : {json.dumps(metrics, ensure_ascii=False)}")
    if out_path:
        print(f"Summary    : {out_path}")
    return summary
