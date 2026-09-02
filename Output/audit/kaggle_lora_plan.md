# Kaggle Tesla T4 Runtime Optimization & GPU Execution Guide

**Document Version:** 1.0.0  
**Date:** 2026-09-02  
**Role:** ML Research Engineer  
**Target Hardware:** Kaggle Cloud — Nvidia Tesla T4 (Turing, 16GB GDDR6 VRAM)  

---

## 1. Hardware Specifications & Profile

| Parameter | Specification | Project Implications |
| :--- | :--- | :--- |
| **GPU Architecture** | Nvidia Turing (TU104) | Full hardware support for mixed precision Tensor Cores. |
| **Total VRAM** | 15,360 MB (~15.0 GB available) | Sufficient for batch size 16-32 at seq_len 128 without OOM. |
| **FP16 Tensor Cores** | 320 Tensor Cores (65 TFLOPS) | Native AMP yields 2.5x speedup over FP32 single precision. |
| **Compute Capability**| SM 7.5 | Fully compatible with PyTorch 2.x cu12x default container. |
| **Host System** | Ubuntu 22.04 LTS, 4 vCPUs, 30GB RAM | Multi-threaded CPU tokenization and DataLoader pin memory. |

---

## 2. VRAM Budget & Memory Allocation Strategy

For `indolem/indobertweet-base-uncased` (110M parameters) with LoRA ($r=16, \alpha=32$):

```
+-------------------------------------------------------------+
|                     TESLA T4 VRAM BUDGET (15,360 MB)        |
+-------------------------------------------------------------+
| [Base Model Weights (FP16)]         :  ~220 MB (Frozen)     |
| [LoRA Adapter Weights + Opt State]  :  ~15 MB (Trainable)   |
| [Optimizer State (AdamW 32-bit)]    :  ~30 MB               |
| [Activation Memory (BS=16, L=128)]  :  ~1,450 MB (FP16 AMP) |
| [PyTorch Context & CUDA Overhead]   :  ~650 MB              |
| ----------------------------------------------------------- |
| TOTAL ESTIMATED WORKING MEMORY      :  ~2,365 MB (~2.3 GB)  |
| AVAILABLE HEADROOM                  :  ~12,995 MB (~12.7 GB)|
+-------------------------------------------------------------+
```

### Key Takeaways from VRAM Profile:
1. **Zero Out-of-Memory (OOM) Risk:** At batch size 16 and sequence length 128, the working memory requirement is only **~2.4 GB**, utilizing less than **16%** of available Tesla T4 capacity.
2. **Gradient Checkpointing is Unnecessary:** Because VRAM headroom is massive (>12 GB), **`gradient_checkpointing` must remain `False`**. Activating gradient checkpointing would needlessly incur a 25-30% computational time penalty from re-computing forward activations during backward passes.
3. **Batch Size Flexibility:** Batch size can comfortably scale up to 32 or 64 without memory exhaustion if throughput optimization is desired.

---

## 3. Mixed Precision & Environment Configuration

### Mandatory Environment Variables
Kaggle dual-T4 instances sometimes route jobs across both GPUs. In sequence classification, multi-GPU `DataParallel` can silently double effective batch size and halve training steps, leading to undertrained models. Enforce single-GPU execution in cell 1:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### PyTorch AMP Activation
In `TrainingArguments`:
```python
training_args = TrainingArguments(
    fp16=True,                          # Enables Nvidia Tensor Core FP16 execution
    dataloader_pin_memory=True,         # Fast page-locked host-to-device memory copy
    dataloader_num_workers=2,           # Multi-threaded background mini-batch loading
    gradient_accumulation_steps=1,      # Direct gradient step per batch
    optim="adamw_torch",                # PyTorch native AdamW
)
```

---

## 4. Checkpoint Strategy & Kaggle Disk Quota

Kaggle kernels impose a strict **20 GB working disk limit** (`/kaggle/working`). Unrestricted saving of full BERT checkpoints (440 MB each) across 5 epochs and 3 seeds will rapidly consume disk space and trigger kernel write errors.

### Checkpoint Invariants:
1. **Save Limit:** Set `save_total_limit=1` in `TrainingArguments`. This ensures only the single most recent checkpoint is preserved during training.
2. **Adapter-Only Final Export:**
   Save only the lightweight LoRA adapter (`adapter_model.safetensors`, ~2.4 MB) rather than full model weights:
   ```python
   trainer.save_model("Output/indobert_lora/seed42/best_model")
   tokenizer.save_pretrained("Output/indobert_lora/seed42/best_model")
   ```
3. **Intermediate Checkpoint Purge:**
   Delete temporary step checkpoints immediately after `trainer.train()` concludes:
   ```python
   import shutil
   shutil.rmtree(checkpoint_dir, ignore_errors=True)
   ```

---

## 5. Kaggle CLI Execution & Resume Strategy

### Push Contract
Always verify kernel slug and run via CLI from project root:
```powershell
# Verify status before pushing
kaggle kernels status emanuelembuaijdak/baseline-b03-indobert

# Push kernel to run on GPU Tesla T4
kaggle kernels push -p temp_kernel/baseline_b03_indobert
```

### Output Retrieval
Retrieve artifacts without opening the web browser:
```powershell
# Pull execution output files
kaggle kernels output emanuelembuaijdak/baseline-b03-indobert -p .kaggle-outputs/b03/
```
