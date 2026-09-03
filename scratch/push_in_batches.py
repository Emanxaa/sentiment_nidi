import subprocess
import os
import json
import time
import argparse
from pathlib import Path

ROOT = Path(r"d:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT")
KERNEL_BASE = ROOT / "temp_kernel"

BATCH_MAP = {
    1: [
        "01_legacy_preprocessing",
        "01_preprocessing_data_baru_v2",
        "06_analisis_topik",
    ],
    2: [
        "02_legacy_model_lstm",
        "03_legacy_model_bilstm",
        "04_legacy_model_indobertweet_lora",
    ],
    3: [
        "02_model_lstm_data_baru_v2_empiris",
        "03_model_bilstm_data_baru_v2_empiris",
        "04_model_indobertweet_lora_data_baru_v2_empiris",
    ],
    4: [
        "02_model_lstm_data_baru_v2_simulasi",
        "03_model_bilstm_data_baru_v2_simulasi",
        "04_model_indobertweet_lora_data_baru_v2_simulasi",
    ],
    5: [
        "05_evaluasi_komparasi_dan_signifikansi",
    ]
}

def push_batch(batch_num):
    dirs = BATCH_MAP.get(batch_num, [])
    print(f"\n{'='*70}")
    print(f"[*] EXECUTING BATCH {batch_num} ({len(dirs)} kernels)")
    print(f"{'='*70}")
    
    results = []
    for k_name in dirs:
        k_dir = KERNEL_BASE / k_name
        meta_file = k_dir / "kernel-metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        slug = meta["id"]
        print(f"\n--> Target Kernel Slug: {slug}")
        print(f"    Directory: {k_name} | GPU: {meta.get('enable_gpu')}")
        
        cmd = ["kaggle", "kernels", "push", "-p", str(k_dir)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        stdout = res.stdout.strip()
        stderr = res.stderr.strip()
        
        print(f"    STDOUT: {stdout}")
        if stderr:
            print(f"    STDERR: {stderr}")
            
        results.append({
            "slug": slug,
            "success": res.returncode == 0,
            "output": stdout or stderr
        })
        print("    [Cooling down for 10s to prevent rate limiting...]")
        time.sleep(10)
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch number (1-5)")
    args = parser.parse_args()
    push_batch(args.batch)
