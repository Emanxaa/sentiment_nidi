"""Orkestrator pipeline kualitas data (PRD_DATA_QUALITY_PIPELINE.md).

Contoh:
  python run_all.py --phase 0            # audit saja
  python run_all.py --phase 3 --dry-run  # anotasi LLM simulasi (tanpa API key)
  python run_all.py --phase all          # 0 -> 5 (harus berurutan)
  python run_all.py --phase 6 --activate-v2  # bangun + aktifkan v2
"""
from __future__ import annotations

import argparse

from quality_pipeline import (
    phase0_audit,
    phase1_preprocess_audit,
    phase2_sampling,
    phase3_llm_annotation,
    phase4_qa,
    phase5_evaluation,
    phase6_retrain,
)

PHASES = {
    "0": phase0_audit.run,
    "1": phase1_preprocess_audit.run,
    "2": phase2_sampling.run,
    "3": phase3_llm_annotation.run,
    "4": phase4_qa.run,
    "5": phase5_evaluation.run,
    "6": phase6_retrain.run,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Data Quality Improvement Pipeline")
    ap.add_argument("--phase", required=True, help="0-6, atau 'all' (0-5) / 'all6' (0-6)")
    ap.add_argument("--dry-run", action="store_true", help="Mode simulasi (Phase 3) tanpa API key")
    ap.add_argument("--model", default=None, help="Model LLM (Phase 3), default sesuai provider")
    ap.add_argument("--provider", default=None, choices=("openai", "gemini"), help="Provider LLM (Phase 3)")
    ap.add_argument("--only-batch", type=int, default=None, help="Hanya batch tertentu (Phase 3)")
    ap.add_argument("--force", action="store_true", help="Timpa batch yang sudah ada (Phase 3)")
    ap.add_argument("--activate-v2", action="store_true", help="Aktifkan v2 (Phase 6)")
    args = ap.parse_args()

    if args.phase == "all":
        order = ["0", "1", "2", "3", "4", "5"]
    elif args.phase == "all6":
        order = ["0", "1", "2", "3", "4", "5", "6"]
    else:
        order = [args.phase]

    for p in order:
        if p not in PHASES:
            raise SystemExit(f"Fase tidak dikenal: {p}. Gunakan 0-6 atau 'all'.")
        print(f"\n========== Phase {p} ==========")
        kwargs = {"dry_run": args.dry_run}
        if p == "3":
            if args.model:
                kwargs["model"] = args.model
            if args.provider:
                kwargs["provider"] = args.provider
            if args.only_batch is not None:
                kwargs["only_batch"] = args.only_batch
            if args.force:
                kwargs["force"] = True
        if p == "6" and args.activate_v2:
            kwargs["activate"] = True
        PHASES[p](**kwargs)


if __name__ == "__main__":
    main()
