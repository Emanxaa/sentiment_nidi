import os
import json
import shutil
from pathlib import Path

ROOT = Path(r"d:\DATA SCIENCE\jokiidin\Thesis-LSTM-IndoBERT")
NB_DIR = ROOT / "notebooks"
KERNEL_BASE = ROOT / "temp_kernel"
KERNEL_BASE.mkdir(parents=True, exist_ok=True)

DATASET_SLUG = "emanuelembuaijdak/thesis-indobert-processed-data"

# Mapping of the 13 notebooks with exact matched titles
KERNEL_SPECS = [
    # Batch 1: Preprocessing & Topic Modeling
    {
        "batch": 1,
        "dir": "01_legacy_preprocessing",
        "nb": "01_legacy_preprocessing.ipynb",
        "slug": "thesis-01-legacy-preprocessing",
        "title": "Thesis 01 Legacy Preprocessing",
        "gpu": False,
    },
    {
        "batch": 1,
        "dir": "01_preprocessing_data_baru_v2",
        "nb": "01_preprocessing_data_baru_v2.ipynb",
        "slug": "thesis-01-preprocessing-data-v2",
        "title": "Thesis 01 Preprocessing Data V2",
        "gpu": False,
    },
    {
        "batch": 1,
        "dir": "06_analisis_topik",
        "nb": "06_analisis_topik.ipynb",
        "slug": "thesis-06-analisis-topik-lda",
        "title": "Thesis 06 Analisis Topik LDA",
        "gpu": False,
    },
    
    # Batch 2: Legacy Models (2 CPU, 1 GPU)
    {
        "batch": 2,
        "dir": "02_legacy_model_lstm",
        "nb": "02_legacy_model_lstm.ipynb",
        "slug": "baseline-b01-lstm",
        "title": "baseline-b01-lstm",
        "gpu": False,
    },
    {
        "batch": 2,
        "dir": "03_legacy_model_bilstm",
        "nb": "03_legacy_model_bilstm.ipynb",
        "slug": "baseline-b02-bilstm",
        "title": "baseline-b02-bilstm",
        "gpu": False,
    },
    {
        "batch": 2,
        "dir": "04_legacy_model_indobertweet_lora",
        "nb": "04_legacy_model_indobertweet_lora.ipynb",
        "slug": "baseline-b03-indobert",
        "title": "baseline-b03-indobert",
        "gpu": True,
    },
    
    # Batch 3: Data V2 Empiris (2 CPU, 1 GPU)
    {
        "batch": 3,
        "dir": "02_model_lstm_data_baru_v2_empiris",
        "nb": "02_model_lstm_data_baru_v2_empiris.ipynb",
        "slug": "thesis-lstm-v2",
        "title": "Thesis LSTM v2",
        "gpu": False,
    },
    {
        "batch": 3,
        "dir": "03_model_bilstm_data_baru_v2_empiris",
        "nb": "03_model_bilstm_data_baru_v2_empiris.ipynb",
        "slug": "thesis-bilstm-v2",
        "title": "Thesis BiLSTM v2",
        "gpu": False,
    },
    {
        "batch": 3,
        "dir": "04_model_indobertweet_lora_data_baru_v2_empiris",
        "nb": "04_model_indobertweet_lora_data_baru_v2_empiris.ipynb",
        "slug": "thesis-lora-p2-task-adaptive-pretraining",
        "title": "Thesis LoRA - P2 Task Adaptive Pretraining",
        "gpu": True,
    },
    
    # Batch 4: Data V2 Simulasi (2 CPU, 1 GPU)
    {
        "batch": 4,
        "dir": "02_model_lstm_data_baru_v2_simulasi",
        "nb": "02_model_lstm_data_baru_v2_simulasi.ipynb",
        "slug": "thesis-02-lstm-v2-simulasi",
        "title": "Thesis 02 LSTM V2 Simulasi",
        "gpu": False,
    },
    {
        "batch": 4,
        "dir": "03_model_bilstm_data_baru_v2_simulasi",
        "nb": "03_model_bilstm_data_baru_v2_simulasi.ipynb",
        "slug": "thesis-03-bilstm-v2-simulasi",
        "title": "Thesis 03 BiLSTM V2 Simulasi",
        "gpu": False,
    },
    {
        "batch": 4,
        "dir": "04_model_indobertweet_lora_data_baru_v2_simulasi",
        "nb": "04_model_indobertweet_lora_data_baru_v2_simulasi.ipynb",
        "slug": "thesis-04-indobert-v2-simulasi",
        "title": "Thesis 04 IndoBERT V2 Simulasi",
        "gpu": True,
    },
    
    # Batch 5: Evaluasi Master (1 CPU)
    {
        "batch": 5,
        "dir": "05_evaluasi_komparasi_dan_signifikansi",
        "nb": "05_evaluasi_komparasi_dan_signifikansi.ipynb",
        "slug": "thesis-05-evaluasi-dan-signifikansi",
        "title": "Thesis 05 Evaluasi dan Signifikansi",
        "gpu": False,
    }
]

PATH_RESOLVER_CODE = """import os
from pathlib import Path

def resolve_path(filename):
    candidates = [
        Path(f"/kaggle/input/thesis-indobert-processed-data/{filename}"),
        Path(f"/kaggle/input/thesis-indobert-processed-data/Data/{filename}"),
        Path(f"Data/{filename}"),
        Path(f"Data/processed/{filename}"),
        Path(f"Data/raw/{filename}"),
        Path(f"Data/simulated/{filename}"),
        Path(f"kamus/{filename}"),
        Path(f"Output/predictions/{filename}"),
        Path(filename),
        Path(f"../{filename}"),
        Path(f"../Data/{filename}"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename
"""

print(f"[*] Preparing {len(KERNEL_SPECS)} Kaggle kernel packages...")

for spec in KERNEL_SPECS:
    k_dir = KERNEL_BASE / spec["dir"]
    k_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load notebook
    src_nb = NB_DIR / spec["nb"]
    with open(src_nb, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    
    # 2. Filter out any previously broken resolver cell
    nb_data["cells"] = [c for c in nb_data["cells"] if "def resolve_path(" not in "".join(c.get("source", []))]
    
    resolver_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in PATH_RESOLVER_CODE.splitlines()]
    }
    # Insert after header cell
    insert_idx = 1 if len(nb_data["cells"]) > 0 else 0
    nb_data["cells"].insert(insert_idx, resolver_cell)
        
    # Save notebook to target directory
    dst_nb = k_dir / spec["nb"]
    with open(dst_nb, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)
        
    # 3. Create kernel-metadata.json
    metadata = {
        "id": f"emanuelembuaijdak/{spec['slug']}",
        "title": spec["title"],
        "code_file": spec["nb"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "true" if spec["gpu"] else "false",
        "enable_internet": "true",
        "dataset_sources": [DATASET_SLUG],
        "competition_sources": [],
        "kernel_sources": []
    }
    if spec["gpu"]:
        metadata["machine_shape"] = "NvidiaTeslaT4"
        
    with open(k_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[+] Prepared (Batch {spec['batch']}) {spec['dir']} -> slug: {metadata['id']} (GPU: {metadata['enable_gpu']})")

print("\n[*] All 13 kernel directories prepared successfully in temp_kernel/")
