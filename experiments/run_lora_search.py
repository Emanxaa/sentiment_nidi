"""
IndoBERTweet-LoRA Hyperparameter Search (Milestone B2 Placeholder)
==================================================================
Systematic grid search across LoRA ranks (r in {8, 16}), alpha,
dropout, and learning rates (2e-5 to 2e-4) evaluated on Validation set.

Usage:
    python experiments/run_lora_search.py [--config configs/indobert_lora.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA Hyperparameter Search (B2)")
    parser.add_argument("--config", type=str, default="configs/indobert_lora.yaml", help="Path to base config YAML")
    args = parser.parse_args()
    print(f"[*] LoRA Hyperparameter Search specification ready. Config: {args.config}")
    print("[!] Implementation scheduled for Milestone B2.")


if __name__ == "__main__":
    main()
