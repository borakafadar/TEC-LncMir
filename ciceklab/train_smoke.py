#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_smoke.py
==============
Smoke test training script for TEC-LncMir on ciceklab data.
Designed to run quickly on a local machine (CPU or small GPU) to
verify the pipeline works end-to-end.

Uses:
  - Only 1 training chunk (chunk_0000)
  - 5 epochs
  - Batch size 4
  - Validates on all 3 validation splits
  - No data augmentation

Usage:
    python train_smoke.py [--device -1]  # CPU
    python train_smoke.py [--device 0]   # GPU 0
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score,
)
from tqdm import tqdm

# Add the parent directory to path so we can import from code/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from code.models.contact import ContactCNN
from code.models.interaction import ModelInteraction
from code.utils import get_tokens, get_tokens_word

# Our data adapter
from data_adapter import (
    load_seq_index,
    CiceklabDataset,
    CiceklabChunkDataset,
    collate_paired_sequences,
)


# ---------------------------------------------------------------------------
# Tokenization helpers (from original TEC-LncMir code)
# ---------------------------------------------------------------------------

def build_kmer_dict(k: int) -> dict:
    """Build the k-mer vocabulary dict for lncRNA encoding."""
    bases = "AUCG"
    vocab = {}
    if k == 1:
        vocab = {"A": 1, "G": 2, "C": 3, "U": 4}
    else:
        idx = 0
        from itertools import product
        for combo in product(bases, repeat=k):
            idx += 1
            vocab[''.join(combo)] = idx
    return vocab


def tokenize_batch(lnc_seqs, mi_seqs, base_number_dict_mirna, base_number_dict_lnc, device):
    """
    Tokenize a batch of lncRNA and miRNA sequences.
    Returns a dict mapping sequence → token tensor.

    Matches the tokenization logic in the original train.py lines 370-377.
    """
    # Deduplicate sequences
    unique_mi = list(set(mi_seqs))
    unique_lnc = list(set(lnc_seqs))

    # Format: [[raw_seq, cleaned_seq], ...]
    mirnas = [[s, s.replace('-', '').replace('>', '').replace('T', 'U')] for s in unique_mi]
    lncrnas = [[s, s.replace('-', '').replace('>', '').replace('T', 'U')] for s in unique_lnc]

    rna_list = get_tokens(mirnas, base_number_dict_mirna) + get_tokens_word(lncrnas, base_number_dict_lnc)

    for i in range(len(rna_list)):
        rna_list[i][1] = torch.LongTensor(rna_list[i][1]).to(device)

    embeddings = {rna_list[i][0]: rna_list[i][1] for i in range(len(rna_list))}
    return embeddings


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(args, model, lnc_seqs, mi_seqs, y, base_number_dict_mirna,
               base_number_dict_lnc, device):
    """Forward + backward pass for one batch."""
    embeddings = tokenize_batch(lnc_seqs, mi_seqs, base_number_dict_mirna,
                                 base_number_dict_lnc, device)

    b = len(lnc_seqs)
    z_a, z_b = [], []
    for i in range(b):
        z_a.append(embeddings[lnc_seqs[i]])
        z_b.append(embeddings[mi_seqs[i]])

    z_a = torch.nn.utils.rnn.pad_sequence(z_a, batch_first=True).reshape(b, -1, 1)
    z_b = torch.nn.utils.rnn.pad_sequence(z_b, batch_first=True).reshape(b, -1, 1)

    c_map_mag, p_hat = model.map_predict(z_a, z_b)

    y = y.to(device)
    y = Variable(y)
    p_hat = p_hat.float()
    bce_loss = F.binary_cross_entropy(p_hat, y.float())
    bce_loss.backward()

    with torch.no_grad():
        p_guess = (p_hat > 0.5).float()

    return bce_loss.item(), p_guess.cpu().int().tolist(), y.cpu().int().tolist()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataset, base_number_dict_mirna, base_number_dict_lnc, device,
             batch_size=4):
    """Evaluate model on a dataset, return metrics dict."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=collate_paired_sequences, shuffle=False,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for lnc_seqs, mi_seqs, y in tqdm(loader, desc="Evaluating", leave=False):
            embeddings = tokenize_batch(lnc_seqs, mi_seqs, base_number_dict_mirna,
                                         base_number_dict_lnc, device)
            b = len(lnc_seqs)
            z_a, z_b = [], []
            for i in range(b):
                z_a.append(embeddings[lnc_seqs[i]])
                z_b.append(embeddings[mi_seqs[i]])

            z_a = torch.nn.utils.rnn.pad_sequence(z_a, batch_first=True).reshape(b, -1, 1)
            z_b = torch.nn.utils.rnn.pad_sequence(z_b, batch_first=True).reshape(b, -1, 1)

            _, p_hat = model.map_predict(z_a, z_b)
            all_preds.extend(p_hat.cpu().float().tolist())
            all_labels.extend(y.int().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    if len(set(labels)) < 2:
        pred_binary = (preds >= 0.5).astype(int)
        return {"accuracy": accuracy_score(labels, pred_binary), "n_samples": len(labels)}

    pred_binary = (preds >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()

    metrics = {
        "n_samples": len(labels),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "mcc": ((tp * tn - fp * fn) /
                (((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn)) ** 0.5)
                if ((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn)) > 0 else 0),
        "auroc": roc_auc_score(labels, preds),
        "aupr": average_precision_score(labels, preds),
    }
    return metrics


def print_metrics(name, metrics):
    """Print evaluation metrics in a clean format."""
    print(f"\n  === {name} ({metrics['n_samples']} samples) ===")
    for k, v in metrics.items():
        if k == 'n_samples':
            continue
        print(f"    {k:>12s}: {v:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TEC-LncMir Smoke Test Training")

    # Data
    parser.add_argument("--seq-index", type=str,
                        default=os.path.join(os.path.dirname(__file__), "seq_index.pkl"),
                        help="Path to seq_index.pkl")
    parser.add_argument("--train-chunks", type=str,
                        default=os.path.join(os.path.dirname(__file__), "training_chunks"),
                        help="Path to training_chunks directory")
    parser.add_argument("--valid-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "data_with_negatives", "rna_rna", "miRNA_lncRNA"),
                        help="Path to miRNA_lncRNA validation/test directory")

    # Smoke test settings
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of training chunks to use (default: 1)")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")

    # Model hyperparameters (paper defaults)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--one-word", type=int, default=4, help="k-mer size for lncRNA (default: 4)")
    parser.add_argument("--ks", type=int, default=1, help="CNN kernel size (default: 1)")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-lnc-len", type=int, default=25000,
                        help="Max lncRNA sequence length (default: 25000)")

    # Output
    parser.add_argument("-o", "--outdir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "output_smoke"),
                        help="Output directory")
    parser.add_argument("-d", "--device", type=int, default=-1,
                        help="Device: -1 for CPU, 0+ for GPU")

    # For compatibility with ModelInteraction constructor
    parser.add_argument("--no-sigmoid", action="store_true")
    parser.add_argument("--p0", type=float, default=0.5)

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device
    if args.device == -1:
        args.device = 'cpu'
        print("Using CPU")
    else:
        args.device = args.device
        print(f"Using CUDA device {args.device} - {torch.cuda.get_device_name(args.device)}")

    # Create output dir
    os.makedirs(args.outdir, exist_ok=True)

    # Load sequence index
    seq_index = load_seq_index(args.seq_index)

    # Load training data (just 1 chunk for smoke test)
    print(f"\nLoading training data (first {args.num_chunks} chunk(s))...")
    from data_adapter import CiceklabMergedDataset
    train_dataset = CiceklabMergedDataset(
        args.train_chunks, seq_index,
        max_lnc_len=args.max_lnc_len,
        max_chunks=args.num_chunks,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=True,
    )
    print(f"Training: {len(train_dataset)} pairs")

    # Load validation datasets
    print("\nLoading validation datasets...")
    valid_datasets = {}
    for split_name in ["final_valid_unseen_pair", "final_valid_unseen_source", "final_valid_unseen_target"]:
        path = os.path.join(args.valid_dir, f"{split_name}.jsonl")
        if os.path.exists(path):
            valid_datasets[split_name] = CiceklabDataset(path, seq_index, args.max_lnc_len)

    # Build k-mer dictionaries
    base_number_dict_mirna = {"A": 1, "G": 2, "C": 3, "U": 4}
    base_number_dict_lnc = build_kmer_dict(args.one_word)

    # Create model (same architecture as paper)
    print("\nInitializing TEC-LncMir model...")
    contact_model = ContactCNN(args.ks, args.projection_dim)
    model = ModelInteraction(
        args,
        contact_model,
        do_sigmoid=not args.no_sigmoid,
        p0=args.p0,
    )
    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0)

    # Training loop
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST TRAINING")
    print(f"  Epochs: {args.num_epochs}, Batch size: {args.batch_size}")
    print(f"  Training pairs: {len(train_dataset)}")
    print(f"{'='*60}\n")

    best_mcc = -1.0
    log_path = os.path.join(args.outdir, "log.txt")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        start_time = time.time()

        for batch_idx, (lnc_seqs, mi_seqs, y) in enumerate(train_loader):
            loss, preds, labels = train_step(
                args, model, lnc_seqs, mi_seqs, y,
                base_number_dict_mirna, base_number_dict_lnc, args.device,
            )
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss * len(lnc_seqs)
            epoch_correct += sum(p == l for p, l in zip(preds, labels))
            epoch_total += len(lnc_seqs)

            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / epoch_total
                avg_acc = epoch_correct / epoch_total
                print(f"  [Epoch {epoch+1}/{args.num_epochs}] "
                      f"Batch {batch_idx+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
        avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        print(f"\n  Epoch {epoch+1}/{args.num_epochs} done in {elapsed:.1f}s: "
              f"loss={avg_loss:.4f}, acc={avg_acc:.4f}")

        # Log
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}, time={elapsed:.1f}s\n")

        # Validate at end of each epoch
        print("\n  Validating...")
        for split_name, vd in valid_datasets.items():
            metrics = evaluate(model, vd, base_number_dict_mirna,
                             base_number_dict_lnc, args.device, args.batch_size)
            print_metrics(split_name, metrics)

            with open(log_path, 'a') as f:
                f.write(f"  {split_name}: {metrics}\n")

            # Save best model based on unseen_pair MCC
            if 'unseen_pair' in split_name and metrics.get('mcc', -1) > best_mcc:
                best_mcc = metrics.get('mcc', -1)
                save_path = os.path.join(args.outdir, "best_model.sav")
                model.cpu()
                torch.save(model, save_path)
                model.to(args.device)
                print(f"\n  >>> New best model! MCC={best_mcc:.4f}, saved to {save_path}")

                with open(os.path.join(args.outdir, "best_metrics.txt"), 'w') as f:
                    f.write(f"Epoch: {epoch+1}\n")
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")

        model.train()
        print()

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST COMPLETE!")
    print(f"  Best MCC: {best_mcc:.4f}")
    print(f"  Output: {args.outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
