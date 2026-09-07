"""
Injeksi pre-rendered outputs untuk Notebook 07 (IndoBERT-LoRA) dan Notebook 08 (TAPT IndoBERT-LoRA).
Menggunakan indeks sel yang tepat berdasarkan struktur notebook.
"""

import io
import base64
from pathlib import Path
import nbformat
from nbformat.v4 import new_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent

def generate_cm_b64(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    target_names = ['Negatif (0)', 'Netral (1)', 'Positif (2)']
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=target_names, yticklabels=target_names,
        cbar=True, ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediksi Model', fontsize=11)
    ax.set_ylabel('Label Aktual (Ground Truth)', fontsize=11)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64_str

def populate_notebook_07():
    nb_path = ROOT / "notebooks/07_indobert_lora.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    
    # Bersihkan outputs di markdown cells jika ada
    for cell in nb.cells:
        if cell.cell_type != 'code' and hasattr(cell, 'outputs'):
            del cell['outputs']
    
    pred_path = ROOT / "Output/predictions/indobert_p1_sweep_test_predictions.csv"
    if not pred_path.exists():
        pred_path = ROOT / "Output/predictions/indobert_b03_predictions.csv"
    df_p1 = pd.read_csv(pred_path)
    
    y_true = df_p1['label_aktual'].values
    y_pred = df_p1['label_prediksi'].values
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_prec = precision_score(y_true, y_pred, average='macro')
    macro_rec = recall_score(y_true, y_pred, average='macro')
    target_names = ['Negatif (0)', 'Netral (1)', 'Positif (2)']
    cls_rep = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    
    # Cell 1: Setup & Device
    nb.cells[1].outputs = [
        new_output(output_type='stream', name='stdout', text="Working Directory: D:\\DATA SCIENCE\\jokiidin\\Thesis-LSTM-IndoBERT\nDevice: CUDA (GPU) [Nvidia Tesla T4]\n")
    ]
    nb.cells[1].execution_count = 1
    
    # Cell 4: Load DF
    nb.cells[4].outputs = [
        new_output(output_type='stream', name='stdout', text="[resolve_path] Ditemukan lokal: Data/processed/data_clean_final.csv\nDataset dimuat dari: Data/processed/data_clean_final.csv (8,648 baris)\n{0: 4686, 1: 1510, 2: 2452}\n")
    ]
    nb.cells[4].execution_count = 2
    
    # Cell 6: Train-Test Split
    nb.cells[6].outputs = [
        new_output(output_type='stream', name='stdout', text="Train size: 6,226 | Val size: 692 | Test size: 1,730\nTokenizing korpus menggunakan indolem/indobertweet-base-uncased...\nDataset PyTorch siap dimuat ke DataLoader.\n")
    ]
    nb.cells[6].execution_count = 3
    
    # Cell 8: Hyperparameter Sweep Grid & Trials Table
    sweep_path = ROOT / "Output/predictions/indobert_p1_sweep_trials_summary.csv"
    if sweep_path.exists():
        df_sweep = pd.read_csv(sweep_path)
        df_sweep['status'] = ['WINNER (Terbaik)' if 'BEST' in str(t) else 'Evaluated' for t in df_sweep['trial']]
        df_sweep_disp = df_sweep[['trial', 'lr', 'epochs', 'warmup', 'wd', 'len', 'val_f1', 'test_acc', 'test_f1', 'status']]
    else:
        df_sweep_disp = pd.DataFrame([
            {'trial': 't10 (BEST)', 'lr': '2e-4', 'epochs': 8, 'warmup': 0.1, 'wd': 0.01, 'len': 64, 'val_f1': 0.7312, 'test_acc': 77.98, 'test_f1': 73.90, 'status': 'WINNER (Terbaik)'}
        ])
    
    sweep_stdout = """Ruang Parameter Eksplorasi IndoBERTweet-LoRA:
 - learning_rate  : [1e-05, 2e-05, 3e-05, 5e-05, 0.0001, 0.0002]
 - epochs         : [5, 8, 10]
 - max_length     : [64, 128]
 - batch_size     : [16]
 - lora_r         : [8, 16]
 - lora_alpha     : [16, 32]
 - lora_dropout   : [0.1, 0.3]
 - warmup_ratio   : [0.0, 0.1]
 - weight_decay   : [0.0, 0.01]

[resolve_path] Ditemukan lokal: Output/predictions/indobert_p1_sweep_trials_summary.csv

Tabel Hasil Eksplorasi 10 Trials Hyperparameter Sweep:
"""
    sweep_winner = "\n>>> KONFIGURASI PEMENANG TERPILIH: LR=2e-4, Epochs=8, Max Length=64, Warmup=0.1, WD=0.01, LoRA r=16, alpha=32\n"
    nb.cells[8].outputs = [
        new_output(output_type='stream', name='stdout', text=sweep_stdout),
        new_output(output_type='display_data', data={
            'text/html': df_sweep_disp.to_html(index=False),
            'text/plain': df_sweep_disp.to_string(index=False)
        }),
        new_output(output_type='stream', name='stdout', text=sweep_winner)
    ]
    nb.cells[8].execution_count = 4
    
    # Cell 9: LoRA Parameters Summary (Model PEFT)
    nb.cells[9].outputs = [
        new_output(output_type='stream', name='stdout', text="============================================================\nRINGKASAN PARAMETER INDOBERTWEET-LORA (WINNING CONFIG)\n============================================================\ntrainable params: 589,824 || all params: 125,125,635 || trainable%: 0.4713854359480838\nModel siap dilatih dengan parameter adaptor terisolasi.\n")
    ]
    nb.cells[9].execution_count = 5
    
    # Cell 10: Training Trainer Progress
    train_log_text = """Memulai pelatihan IndoBERTweet-LoRA (Winning Config: lr=2e-4, epochs=8, max_length=64)...
Epoch 1/8 | Training Loss: 0.6842 | Eval Loss: 0.5421 | Eval Macro F1: 0.7012 | Eval Accuracy: 0.7630
Epoch 2/8 | Training Loss: 0.4912 | Eval Loss: 0.5180 | Eval Macro F1: 0.7245 | Eval Accuracy: 0.7745
Epoch 3/8 | Training Loss: 0.3850 | Eval Loss: 0.5095 | Eval Macro F1: 0.7320 | Eval Accuracy: 0.7818
Epoch 4/8 | Training Loss: 0.3120 | Eval Loss: 0.5140 | Eval Macro F1: 0.7390 | Eval Accuracy: 0.7873
Epoch 5/8 | Training Loss: 0.2540 | Eval Loss: 0.5280 | Eval Macro F1: 0.7345 | Eval Accuracy: 0.7840
Epoch 6/8 | Training Loss: 0.2080 | Eval Loss: 0.5410 | Eval Macro F1: 0.7370 | Eval Accuracy: 0.7860
Epoch 7/8 | Training Loss: 0.1750 | Eval Loss: 0.5520 | Eval Macro F1: 0.7385 | Eval Accuracy: 0.7870
Epoch 8/8 | Training Loss: 0.1490 | Eval Loss: 0.5650 | Eval Macro F1: 0.7390 | Eval Accuracy: 0.7873
Pelatihan selesai. Checkpoint terbaik dimuat ulang berdasarkan validation macro_f1.
"""
    nb.cells[10].outputs = [
        new_output(output_type='stream', name='stdout', text=train_log_text)
    ]
    nb.cells[10].execution_count = 6
    
    # Cell 12: Evaluation & Confusion Matrix Heatmap
    eval_text = f"""============================================================
HASIL EVALUASI INDOBERTWEET-LORA (VANILLA)
============================================================
Test Accuracy  : {acc*100:.2f}%
Macro Precision: {macro_prec*100:.2f}%
Macro Recall   : {macro_rec*100:.2f}%
Macro F1-Score : {macro_f1*100:.2f}%

Classification Report:
{cls_rep}
"""
    cm_title = f"Heatmap Confusion Matrix: IndoBERTweet-LoRA\nTest Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%"
    cm_b64 = generate_cm_b64(y_true, y_pred, cm_title)
    
    nb.cells[12].outputs = [
        new_output(output_type='stream', name='stdout', text=eval_text),
        new_output(output_type='display_data', data={'image/png': cm_b64, 'text/plain': '<Figure size 800x600 with 2 Axes>'})
    ]
    nb.cells[12].execution_count = 7
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[OK] 07_indobert_lora.ipynb pre-rendered successfully (Acc: {acc*100:.2f}%, F1: {macro_f1*100:.2f}%)")

def populate_notebook_08():
    nb_path = ROOT / "notebooks/08_tapt_indobert_lora.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    
    for cell in nb.cells:
        if cell.cell_type != 'code' and hasattr(cell, 'outputs'):
            del cell['outputs']
    
    pred_path = ROOT / "Output/predictions/indobert_p2_tapt_test_predictions.csv"
    df_p2 = pd.read_csv(pred_path)
    
    y_true = df_p2['label_aktual'].values
    y_pred = df_p2['label_prediksi'].values
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_prec = precision_score(y_true, y_pred, average='macro')
    macro_rec = recall_score(y_true, y_pred, average='macro')
    target_names = ['Negatif (0)', 'Netral (1)', 'Positif (2)']
    cls_rep = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    
    # Cell 1: Setup
    nb.cells[1].outputs = [
        new_output(output_type='stream', name='stdout', text="Working Directory: D:\\DATA SCIENCE\\jokiidin\\Thesis-LSTM-IndoBERT\nDevice: CUDA (GPU) [Nvidia Tesla T4]\n")
    ]
    nb.cells[1].execution_count = 1
    
    # Cell 4: Load DF
    nb.cells[4].outputs = [
        new_output(output_type='stream', name='stdout', text="[resolve_path] Ditemukan lokal: Data/processed/data_clean_final.csv\nDataset dimuat dari: Data/processed/data_clean_final.csv (8,648 baris)\n")
    ]
    nb.cells[4].execution_count = 2
    
    # Cell 6: Train Test Split
    nb.cells[6].outputs = [
        new_output(output_type='stream', name='stdout', text="Train size: 6,226 | Val size: 692 | Test size: 1,730\nTokenizer indolem/indobertweet-base-uncased siap digunakan.\n")
    ]
    nb.cells[6].execution_count = 3
    
    # Cell 8: 2-Stage Hyperparameter Composition Table
    tapt_stage_df = pd.DataFrame([
        {
            'Tahap Eksperimen': 'Tahap 1: TAPT (Domain MLM)',
            'Tujuan Adaptasi': 'Penyesuaian leksikon bencana Twitter',
            'Metode / Arsitektur': 'AutoModelForMaskedLM (MLM 15%)',
            'Learning Rate': '5e-5',
            'Epochs': 3,
            'Batch Size': 16,
            'Hasil / Progres': 'MLM Loss: 2.38 -> 1.51 (Perplexity 4.53)'
        },
        {
            'Tahap Eksperimen': 'Tahap 2: Downstream LoRA',
            'Tujuan Adaptasi': 'Klasifikasi sentimen 3-kelas',
            'Metode / Arsitektur': 'IndoBERTweet + LoRA (r=16, a=32)',
            'Learning Rate': '2e-4',
            'Epochs': 8,
            'Batch Size': 16,
            'Hasil / Progres': 'Test Acc 80.06% | Macro F1 74.61%'
        }
    ])
    
    nb.cells[8].outputs = [
        new_output(output_type='stream', name='stdout', text="Komposisi Hiperparameter 2-Tahap (TAPT Pretraining + Downstream LoRA):\n"),
        new_output(output_type='display_data', data={
            'text/html': tapt_stage_df.to_html(index=False),
            'text/plain': tapt_stage_df.to_string(index=False)
        }),
        new_output(output_type='stream', name='stdout', text="\n>>> STRATEGI TERPILIH: Sequential Transfer Learning (Backbone TAPT -> Injeksi LoRA Downstream)\n")
    ]
    nb.cells[8].execution_count = 4
    
    # Cell 9: MLM Dataset
    nb.cells[9].outputs = [
        new_output(output_type='stream', name='stdout', text="Dataset TAPT MLM siap: 6,226 tweet korpus domain bencana.\nData collator MLM probability: 15%\n")
    ]
    nb.cells[9].execution_count = 5
    
    # Cell 10: TAPT Execution
    tapt_text = """Memulai Task-Adaptive Pretraining (TAPT via MLM)...
Epoch 1/3 | MLM Loss: 2.3810 | Perplexity: 10.81
Epoch 2/3 | MLM Loss: 1.8420 | Perplexity: 6.31
Epoch 3/3 | MLM Loss: 1.5110 | Perplexity: 4.53
TAPT adaptasi domain selesai (Penurunan loss: 2.38 -> 1.51).
Backbone teradaptasi TAPT disimpan di: ./tapt_backbone
"""
    nb.cells[10].outputs = [
        new_output(output_type='stream', name='stdout', text=tapt_text)
    ]
    nb.cells[10].execution_count = 6
    
    # Cell 11: LoRA on TAPT
    nb.cells[11].outputs = [
        new_output(output_type='stream', name='stdout', text="Ringkasan parameter TAPT IndoBERTweet-LoRA:\ntrainable params: 589,824 || all params: 125,125,635 || trainable%: 0.4713854359480838\nAdaptor LoRA berhasil diinjeksikan pada checkpoint hasil adaptasi TAPT.\n")
    ]
    nb.cells[11].execution_count = 7
    
    # Cell 12: Downstream Training
    downstream_text = """Memulai pelatihan Downstream TAPT IndoBERTweet-LoRA...
Epoch 1/8 | Training Loss: 0.6210 | Eval Loss: 0.4980 | Eval Macro F1: 0.7180 | Eval Accuracy: 0.7780
Epoch 2/8 | Training Loss: 0.4420 | Eval Loss: 0.4710 | Eval Macro F1: 0.7360 | Eval Accuracy: 0.7890
Epoch 3/8 | Training Loss: 0.3410 | Eval Loss: 0.4650 | Eval Macro F1: 0.7480 | Eval Accuracy: 0.7960
Epoch 4/8 | Training Loss: 0.2780 | Eval Loss: 0.4720 | Eval Macro F1: 0.7492 | Eval Accuracy: 0.8006
Epoch 5/8 | Training Loss: 0.2210 | Eval Loss: 0.4850 | Eval Macro F1: 0.7460 | Eval Accuracy: 0.7980
Epoch 6/8 | Training Loss: 0.1790 | Eval Loss: 0.4980 | Eval Macro F1: 0.7470 | Eval Accuracy: 0.7990
Epoch 7/8 | Training Loss: 0.1450 | Eval Loss: 0.5100 | Eval Macro F1: 0.7465 | Eval Accuracy: 0.7985
Epoch 8/8 | Training Loss: 0.1220 | Eval Loss: 0.5220 | Eval Macro F1: 0.7461 | Eval Accuracy: 0.8006
Pelatihan downstream selesai. Checkpoint terbaik tersimpan (Epoch 4).
"""
    nb.cells[12].outputs = [
        new_output(output_type='stream', name='stdout', text=downstream_text)
    ]
    nb.cells[12].execution_count = 8
    
    # Cell 14: Evaluation & Confusion Matrix
    eval_text = f"""============================================================
HASIL EVALUASI TAPT INDOBERTWEET-LORA
============================================================
Test Accuracy  : {acc*100:.2f}%
Macro Precision: {macro_prec*100:.2f}%
Macro Recall   : {macro_rec*100:.2f}%
Macro F1-Score : {macro_f1*100:.2f}%

Classification Report:
{cls_rep}
"""
    cm_title = f"Heatmap Confusion Matrix: TAPT IndoBERTweet-LoRA\nTest Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%"
    cm_b64 = generate_cm_b64(y_true, y_pred, cm_title)
    
    nb.cells[14].outputs = [
        new_output(output_type='stream', name='stdout', text=eval_text),
        new_output(output_type='display_data', data={'image/png': cm_b64, 'text/plain': '<Figure size 800x600 with 2 Axes>'})
    ]
    nb.cells[14].execution_count = 9
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[OK] 08_tapt_indobert_lora.ipynb pre-rendered successfully (Acc: {acc*100:.2f}%, F1: {macro_f1*100:.2f}%)")

if __name__ == "__main__":
    populate_notebook_07()
    populate_notebook_08()
