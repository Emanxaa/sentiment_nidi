# Kaggle Tesla T4 Resource Allocation & Execution Blueprint

**Document Version:** 1.0.0  
**Date:** 2026-09-02  
**Role:** ML Research Engineer  
**Scope:** Hardware Resource Management, VRAM Profiling, and Quota Preservation  

---

## 1. Hardware Architecture & Capacity

All transformer experiments target the **Kaggle Cloud Tesla T4** environment:
* **GPU Accelerator:** Nvidia Tesla T4 (Turing Architecture, TU104).
* **VRAM Capacity:** 16,384 MB total (15,360 MB allocatable in container).
* **Tensor Cores:** 320 Second-Generation Tensor Cores (optimized for FP16 Matrix Multiply-Accumulate).
* **Compute Capability:** 7.5.
* **Host Environment:** 4 Intel Xeon vCPUs, 30 GB System RAM, 20 GB ephemeral disk storage (`/kaggle/working`).

---

## 2. VRAM Profiling & Memory Footprint

Detailed memory allocation breakdown during IndoBERTweet-LoRA training at **batch size 16** and **sequence length 128**:

| Component | Precision | Size in Memory | Notes & Allocation Mechanism |
| :--- | :---: | :---: | :--- |
| **Base Model Weights** | FP16 | ~220 MB | Pretrained `indobertweet-base-uncased` (110.56M weights, frozen). |
| **LoRA Adapter Weights** | FP32 / FP16 | ~15 MB | Low-rank matrices $A$ and $B$ for attention projection layers (~592k weights). |
| **Optimizer States** | FP32 | ~35 MB | First and second moments ($m_t, v_t$) tracked strictly for trainable weights. |
| **Activation Memory** | FP16 | ~1,450 MB | Forward-pass intermediate activations stored for backpropagation. |
| **CUDA / Context Overhead**| — | ~650 MB | PyTorch CUDA runtime, memory pools, and communication buffers. |
| **TOTAL WORKING MEMORY** | — | **~2,370 MB (~2.37 GB)** | **Utilizes only 15.4% of available Tesla T4 VRAM.** |
| **AVAILABLE HEADROOM** | — | **~12,990 MB (~12.63 GB)** | **Massive safety margin against OOM errors.** |

---

## 3. Engineering Recommendations

### A. Mixed Precision (`fp16=True`)
* **Recommendation:** **MANDATORY ENABLED.**
* **Technical Justification:** Tesla T4 Turing Tensor Cores deliver 65 TFLOPS of half-precision throughput compared to only 8.1 TFLOPS in single precision (FP32). Enabling native AMP reduces training step latency by **~2.4x** and cuts activation memory by **50%** with zero degradation in classification precision.

### B. Gradient Checkpointing
* **Recommendation:** **MANDATORY DISABLED (`gradient_checkpointing=False`).**
* **Technical Justification:** Gradient checkpointing discards forward activations and recomputes them during backpropagation to trade compute for memory. Because VRAM headroom is massive (>12.6 GB available), enabling gradient checkpointing would introduce a **25% to 30% compute penalty** for zero practical benefit.

### C. Checkpoint Frequency & Disk Storage Management
* **Kaggle Invariant:** Kaggle kernels enforce a hard **20 GB disk limit**. A full PyTorch sequence classification checkpoint occupies ~440 MB. Saving checkpoints every epoch across 3 seeds would consume $>6.6\text{ GB}$ of disk space, risking silent kernel crashes.
* **Preservation Policy:**
  1. Set `save_total_limit=1` in `TrainingArguments` to retain only the single best evaluation checkpoint.
  2. Set `load_best_model_at_end=True` with `metric_for_best_model="macro_f1"`.
  3. Export **only the PEFT adapter weights** (`adapter_model.safetensors`, ~2.4 MB) and tokenizer configuration to `best_model/`.
  4. Purge intermediate optimizer checkpoint folders (`checkpoint-xxx`) immediately following training completion.

### D. Multi-GPU Guard & Threading
* Always force single GPU binding to prevent Kaggle dual-GPU instances from splitting batches via `DataParallel`:
  ```python
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  ```

---

## 4. Headless Resume & Kaggle CLI Workflow

To guarantee 100% reproducible execution independent of web browser sessions:

1. **Push Kernel:**
   ```powershell
   kaggle kernels push -p temp_kernel/baseline_b03_indobert
   ```
2. **Monitor Status:**
   ```powershell
   kaggle kernels status emanuelembuaijdak/baseline-b03-indobert
   ```
3. **Automated Artifact Retrieval:**
   ```powershell
   kaggle kernels output emanuelembuaijdak/baseline-b03-indobert -p Output/indobert_lora/
   ```
