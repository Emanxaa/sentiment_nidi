"""
IndoBERTweet-LoRA Pipeline Entrypoint
====================================
Unified runner for baseline (B1) and balancing experiments (B3: Class Weight,
B4: ROS, B5: RUS, B6: SMOTE) with 3-seed reproducibility and full reporting.

Usage:
    python experiments/run_indobert_lora.py [--config configs/indobert_lora.yaml] [--seeds 42 123 456]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_b1_indobert import run_three_seed_indobert


def main() -> None:
    parser = argparse.ArgumentParser(description="IndoBERTweet-LoRA Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/indobert_lora.yaml", help="Path to config YAML")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Seeds to evaluate")
    args = parser.parse_args()

    print("[*] Launching IndoBERTweet-LoRA pipeline...")
    run_three_seed_indobert(config_path=args.config, seeds=args.seeds)


if __name__ == "__main__":
    main()
