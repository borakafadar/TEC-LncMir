# TEC-LncMir on Ciceklab Data — Guide

This directory contains scripts to train and evaluate the **TEC-LncMir** model on the **ciceklab lncRNA-miRNA interaction dataset**. The model architecture is taken directly from the paper:

> Tingpeng Yang, Yonghong He, Yu Wang. *Introducing TEC-LncMir for prediction of lncRNA-miRNA interactions through deep learning of RNA sequences.* Briefings in Bioinformatics, 2025.

---

## What We Changed / Added

### New Files (in `ciceklab/`)

| File | Purpose |
|------|---------|
| `build_seq_index.py` | One-time script to scan `rna.fa` and create a fast ID→sequence lookup dictionary |
| `data_adapter.py` | PyTorch Dataset adapters that convert JSONL data to the format TEC-LncMir expects |
| `train_smoke.py` | Smoke test training — 1 chunk, 5 epochs, for local verification |
| `train_full.py` | Full training — all chunks, 300 epochs, for GPU servers |
| `evaluate_ciceklab.py` | Evaluation on all 3 test splits with comprehensive metrics and plots |
| `GUIDE.md` | This documentation file |

### What We Did NOT Change

The **model architecture code** in `code/models/` is **used as-is** from the original paper:
- `code/models/contact.py` — 4-layer ContactCNN
- `code/models/interaction.py` — ModelInteraction with LogisticActivation
- `code/models/transform.py` — EmbeddingTransform with PositionalEncoder + TransformerEncoder
- `code/utils.py` — Tokenization utilities (`get_tokens`, `get_tokens_word`)

We import these directly rather than copying or modifying them.

---

## Key Assumptions

### 1. Data Format Mapping
The original TEC-LncMir code expects:
- A **tab-separated pair file** with columns: `lncRNA_id`, `miRNA_id`, `label`
- Separate **FASTA files** for lncRNA and miRNA sequences

Our ciceklab data uses:
- **JSONL files** with fields: `RNA_id`, `target_id`, `RNA_type`, `target_RNA_type`, `interaction_label`
- A single shared **`rna.fa`** (1.8 GB, 12.3M records) containing all RNA sequences

The `data_adapter.py` bridges this gap.

### 2. Pair Directionality
The JSONL records can have the miRNA-lncRNA pair in **either order**:
- `RNA_type=miRNA, target_RNA_type=lncRNA` → `(RNA_id=miRNA, target_id=lncRNA)`
- `RNA_type=lncRNA, target_RNA_type=miRNA` → `(RNA_id=lncRNA, target_id=miRNA)`

Our adapter handles both orderings and always returns `(lncRNA_seq, miRNA_seq, label)`.

### 3. ID→Sequence Resolution
Rather than relying on regex patterns to filter `rna.fa`, `build_seq_index.py` first scans all training chunks and evaluation datasets (`training_chunks/*.jsonl` and `data_with_negatives/rna_rna/miRNA_lncRNA/*.jsonl`) to collect the exact set of required miRNA and lncRNA IDs. It then performs a single streaming pass over `rna.fa` to extract only those exact sequences into `seq_index.pkl`. Any pairs whose ID is not found in `rna.fa` are logged and silently skipped.

### 4. lncRNA Length Cutoff
Following the paper, lncRNA sequences longer than **25,000 bases** are excluded due to GPU memory constraints.

### 5. Filtering for lncRNA-miRNA Only
The training chunks contain multiple interaction types (RNA-RNA, RNA-protein, miRNA-mRNA, etc.). We filter to keep **only miRNA↔lncRNA pairs** (`RNA_type` and `target_RNA_type` must be one of each).

### 6. Streaming vs. In-Memory Training
The original paper loads all pairs into memory. With ~1.4M training pairs across 346 chunks, we **stream chunks one-at-a-time** to avoid excessive memory usage. Chunks are shuffled each epoch, and pairs within each chunk are also shuffled.

### 7. Model Hyperparameters
All hyperparameters match the paper's final configuration (Table from Section "TEC-LncMir training"):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d0` | 128 | Transformer Encoder dimension |
| `d1` | 64 | Projection/scaler output dimension |
| `k` | 4 | lncRNA k-mer size |
| miRNA encoding | 1-mer | Individual base encoding |
| `nh` | 1 | Attention heads |
| `nl` | 4 | Transformer Encoder layers |
| `ks` | 1 | CNN kernel size |
| `lr` | 0.0001 | Learning rate (Adam) |
| `batch_size` | 64 (default) / 16 (paper) | Larger batch size recommended for GPU saturation |
| `epochs` | 10 (default) / 300 (paper) | 10 epochs on 1.4M pairs ≈ 14M samples (far exceeds 300 epochs on paper benchmark datasets) |
| `dropout` | 0.0 | Dropout probability |
| `seed` | 1234 | Random seed |

---

## How to Run

### Prerequisites

Activate the TEC-LncMir conda environment:

```bash
conda activate TEC_LncMir
```

Or ensure you have these key packages:
- Python 3.8+
- PyTorch 1.13+
- BioPython
- scikit-learn
- numpy, pandas, matplotlib, tqdm, einops

### Step 1: Build the Sequence Index (One-Time)

This scans `rna.fa` (1.8GB) and creates `seq_index.pkl` (~10-20MB). **Only needs to be run once.**

```bash
cd ciceklab
python build_seq_index.py
```

This takes ~30-60 seconds and outputs:
- `ciceklab/seq_index.pkl` — the ID→sequence dictionary

You'll see a summary showing how many miRNA and lncRNA sequences were indexed.

### Step 2a: Smoke Test (Local Machine)

Run a quick test to verify everything works:

```bash
# CPU (no GPU required)
python train_smoke.py --device -1

# Or with a GPU
python train_smoke.py --device 0
```

**What it does:**
- Loads 1 training chunk (~8K miRNA-lncRNA pairs)
- Trains for 5 epochs with batch size 4
- Validates on all 3 validation splits after each epoch
- Saves best model to `ciceklab/output_smoke/best_model.sav`

**Expected runtime:** ~10-30 minutes on CPU, ~5-10 minutes on GPU.

### Step 2b: Full Training (GPU Server)

```bash
# Recommended training for Titan RTX / Modern GPU (FP16 AMP + Batch Size 64)
python train_full.py --device 0 --amp --batch-size 64 --num-epochs 10

# With data augmentation
python train_full.py --device 0 --amp --batch-size 64 --num-epochs 10 --augment

# Custom settings
python train_full.py \
    --device 0 \
    --num-epochs 10 \
    --batch-size 64 \
    --lr 0.0001 \
    --amp \
    --val-every-n-chunks 173 \
    --save-every-n-epochs 2

# Resume from checkpoint
python train_full.py --device 0 --checkpoint output_full/checkpoint_epoch2.pt
```

**What it does:**
- Streams through all 346 training chunks per epoch
- Validates periodically (every 50 chunks by default) and at end of each epoch
- Saves best model based on MCC (on unseen pair validation set)
- Saves periodic checkpoints for resume capability

**Output:**
- `ciceklab/output_full/best_model.sav` — best model weights
- `ciceklab/output_full/best_metrics.txt` — metrics of best model
- `ciceklab/output_full/config.json` — training configuration
- `ciceklab/output_full/log.txt` — training log
- `ciceklab/output_full/checkpoint_epoch*.pt` — periodic checkpoints

### Step 3: Evaluate on Test Data

```bash
# Evaluate smoke test model
python evaluate_ciceklab.py \
    --model output_smoke/best_model.sav \
    --device -1

# Evaluate full training model
python evaluate_ciceklab.py \
    --model output_full/best_model.sav \
    --device 0
```

**What it does:**
- Loads the trained model
- Evaluates on 3 test splits:
  - `final_test_unseen_pair.jsonl` (1,224 pairs — unseen pair combinations)
  - `final_test_unseen_source.jsonl` (10,798 pairs — unseen miRNAs)
  - `final_test_unseen_target.jsonl` (13,712 pairs — unseen lncRNAs)
- Generates ROC curves, PR curves, prediction distributions
- Saves a combined `evaluation_results.json`

**Output:**
- `ciceklab/output_eval/evaluation_results.json`
- `ciceklab/output_eval/*_predictions.tsv`
- `ciceklab/output_eval/*_roc.png` / `*_pr.png` / `*_pred_dist.png`

---

## Data Layout

```
ciceklab/
├── rna.fa                           # Shared RNA sequences (1.8 GB)
├── seq_index.pkl                    # Generated: ID→sequence lookup
├── training_chunks/
│   ├── chunk_0000.jsonl             # Training data chunk 0
│   ├── chunk_0001.jsonl             # Training data chunk 1
│   ├── ...
│   └── chunk_0345.jsonl             # Training data chunk 345
├── data_with_negatives/
│   └── rna_rna/
│       └── miRNA_lncRNA/
│           ├── final_train.jsonl            # (NOT USED — we use training_chunks instead)
│           ├── final_valid_unseen_pair.jsonl     # Validation: unseen pairs
│           ├── final_valid_unseen_source.jsonl   # Validation: unseen miRNAs
│           ├── final_valid_unseen_target.jsonl   # Validation: unseen lncRNAs
│           ├── final_test_unseen_pair.jsonl      # Test: unseen pairs
│           ├── final_test_unseen_source.jsonl    # Test: unseen miRNAs
│           └── final_test_unseen_target.jsonl    # Test: unseen lncRNAs
├── build_seq_index.py
├── data_adapter.py
├── train_smoke.py
├── train_full.py
├── evaluate_ciceklab.py
└── GUIDE.md
```

### Important Note on Training Data

Per `notes.md`: We use `training_chunks/` (the sampled training data) for training — **NOT** `final_train.jsonl` in `data_with_negatives`. The chunked data was pre-sampled with specific modality balancing and appearance limits as documented in `training_chunks/generation_stats.json`.

---

## Evaluation Metrics

We report the same metrics as the paper:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct prediction rate |
| **Sensitivity** | True positive rate (recall) |
| **Specificity** | True negative rate |
| **PPV** | Positive predictive value (precision) |
| **NPV** | Negative predictive value |
| **F1 Score** | Harmonic mean of PPV and sensitivity |
| **MCC** | Matthews correlation coefficient (primary metric for model selection) |
| **AUROC** | Area under the ROC curve |
| **AUPR** | Area under the precision-recall curve |

---

## Troubleshooting

### "seq_index.pkl not found"
Run `python build_seq_index.py` first. This only needs to be done once.

### "CUDA out of memory"
- Reduce `--batch-size` (try 8 or 4)
- The 25kb lncRNA cutoff is already applied by default
- Very long lncRNAs create large contact tensors — the cutoff should prevent most OOM issues

### Many pairs skipped ("no-seq")
Some IDs in the JSONL data may not have matching sequences in `rna.fa`. This is expected — the adapter logs how many pairs were skipped. Check the sequence index summary to see coverage.

### Slow training on CPU
The smoke test is designed for CPU but may still be slow for long lncRNA sequences. Consider reducing `--num-chunks` or using a GPU if available.
