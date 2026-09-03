import subprocess
import os
import json
import time
from pathlib import Path

ROOT = Path(r"d:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT")
KERNEL_BASE = ROOT / "temp_kernel"

# Ordered list of directories to push
KERNEL_DIRS = [
    "01_legacy_preprocessing",
    "01_preprocessing_data_baru_v2",
    "02_legacy_model_lstm",
    "02_model_lstm_data_baru_v2_empiris",
    "02_model_lstm_data_baru_v2_simulasi",
    "03_legacy_model_bilstm",
    "03_model_bilstm_data_baru_v2_empiris",
    "03_model_bilstm_data_baru_v2_simulasi",
    "04_legacy_model_indobertweet_lora",
    "04_model_indobertweet_lora_data_baru_v2_empiris",
    "04_model_indobertweet_lora_data_baru_v2_simulasi",
    "05_evaluasi_komparasi_dan_signifikansi",
    "06_analisis_topik",
]

results = []

for k_dir_name in KERNEL_DIRS:
    k_dir = KERNEL_BASE / k_dir_name
    meta_path = k_dir / "kernel-metadata.json"
    if not meta_path.exists():
        print(f"[!] Warning: {meta_path} does not exist. Skipping.")
        continue
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    slug = meta["id"]
    print(f"\n=======================================================")
    print(f"[*] Target Kernel Slug: {slug}")
    print(f"[*] Path: {k_dir}")
    print(f"[*] Code File: {meta['code_file']} | GPU: {meta.get('enable_gpu', 'false')}")
    print(f"=======================================================")
    
    cmd = ["kaggle", "kernels", "push", "-p", str(k_dir)]
    print(f"Executing: {' '.join(cmd)}")
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    stdout = res.stdout.strip()
    stderr = res.stderr.strip()
    
    print(f"STDOUT:\n{stdout}")
    if stderr:
        print(f"STDERR:\n{stderr}")
        
    results.append({
        "slug": slug,
        "dir": k_dir_name,
        "success": res.returncode == 0,
        "output": stdout or stderr
    })
    time.sleep(2)

print("\n\n" + "="*80)
print("PUSH SUMMARY RESULTS:")
print("="*80)
for r in results:
    status = "SUCCESS" if r["success"] else "FAILED"
    print(f"[{status}] {r['slug']} ({r['dir']}) -> {r['output']}")
