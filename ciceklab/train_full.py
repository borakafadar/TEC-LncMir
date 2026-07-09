#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_full.py
=============
Full production training script for TEC-LncMir on ciceklab data.
Designed for a remote server with GPU(s).

Features:
  - Streams through all 346 training chunks per epoch
  - 300 epochs (paper default)
  - Validates on all 3 validation splits periodically
  - Best model selection based on MCC
  - Optional data augmentation (k-mer offset shifting)
  - Checkpoint save/resume
  - Full logging

Usage:
    # Full training on GPU 0
    python train_full.py --device 0

    # Resume from checkpoint
    python train_full.py --device 0 --checkpoint output_full/checkpoint_epoch50.pt

    # With data augmentation
    python train_full.py --device 0 --augment

    # Custom settings
    python train_full.py --device 0 --num-epochs 300 --batch-size 16 --lr 0.0001
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score,
)
from tqdm import tqdm

# Add parent directory to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from code.models.contact import ContactCNN
from code.models.interaction import ModelInteraction
from code.utils import get_tokens, get_tokens_word

from data_adapter import (
    load_seq_index,
    CiceklabDataset,
    CiceklabChunkDataset,
    collate_paired_sequences,
    iter_training_chunks,
)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def build_kmer_dict(k: int) -> dict:
    """Build the k-mer vocabulary dict for lncRNA encoding."""
    bases = "AUCG"
    vocab = {}
    if k == 1:
        vocab = {"A": 1, "G": 2, "C": 3, "U": 4}
    else:
        from itertools import product
        idx = 0
        for combo in product(bases, repeat=k):
            idx += 1
            vocab[''.join(combo)] = idx
    return vocab


def tokenize_batch(lnc_seqs, mi_seqs, base_number_dict_mirna, base_number_dict_lnc, device):
    """Tokenize a batch of lncRNA and miRNA sequences."""
    unique_mi = list(set(mi_seqs))
    unique_lnc = list(set(lnc_seqs))

    mirnas = [[s, s.replace('-', '').replace('>', '').replace('T', 'U')] for s in unique_mi]
    lncrnas = [[s, s.replace('-', '').replace('>', '').replace('T', 'U')] for s in unique_lnc]

    rna_list = get_tokens(mirnas, base_number_dict_mirna) + get_tokens_word(lncrnas, base_number_dict_lnc)

    for i in range(len(rna_list)):
        rna_list[i][1] = torch.LongTensor(rna_list[i][1]).to(device)

    embeddings = {rna_list[i][0]: rna_list[i][1] for i in range(len(rna_list))}
    return embeddings


def augment_lncrna_seq(seq: str, k: int = 4) -> list:
    """
    Data augmentation: generate k versions of the lncRNA sequence
    by shifting the k-mer reading frame.
    Returns list of (original_or_shifted_seq, shift_offset).
    """
    versions = [seq]  # offset 0 = original
    for offset in range(1, k):
        versions.append(seq[offset:])
    return versions


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(args, model, lnc_seqs, mi_seqs, y,
               base_number_dict_mirna, base_number_dict_lnc, device):
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
             batch_size=16):
    """Evaluate model on a dataset, return metrics dict."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=collate_paired_sequences, shuffle=False,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for lnc_seqs, mi_seqs, y in loader:
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
    denom = ((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))

    metrics = {
        "n_samples": len(labels),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "mcc": (tp * tn - fp * fn) / (denom ** 0.5) if denom > 0 else 0,
        "auroc": roc_auc_score(labels, preds),
        "aupr": average_precision_score(labels, preds),
    }
    return metrics


def print_metrics(name, metrics, log_file=None):
    """Print and optionally log evaluation metrics."""
    lines = [f"\n  === {name} ({metrics['n_samples']} samples) ==="]
    for k, v in metrics.items():
        if k == 'n_samples':
            continue
        lines.append(f"    {k:>12s}: {v:.4f}")
    text = '\n'.join(lines)
    print(text)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(text + '\n')


# ---------------------------------------------------------------------------
# Augmented chunk dataset
# ---------------------------------------------------------------------------

class AugmentedChunkDataset(torch.utils.data.Dataset):
    """Wraps a CiceklabChunkDataset and applies data augmentation."""

    def __init__(self, base_dataset, k=4):
        self.augmented_pairs = []
        for i in range(len(base_dataset)):
            lnc_seq, mi_seq, label = base_dataset[i]
            versions = augment_lncrna_seq(lnc_seq, k)
            for v in versions:
                self.augmented_pairs.append((v, mi_seq, label))

    def __len__(self):
        return len(self.augmented_pairs)

    def __getitem__(self, i):
        return self.augmented_pairs[i]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TEC-LncMir Full Training")

    # Data
    parser.add_argument("--seq-index", type=str,
                        default=os.path.join(os.path.dirname(__file__), "seq_index.pkl"))
    parser.add_argument("--train-chunks", type=str,
                        default=os.path.join(os.path.dirname(__file__), "training_chunks"))
    parser.add_argument("--valid-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "data_with_negatives", "rna_rna", "miRNA_lncRNA"))

    # Training settings
    parser.add_argument("--num-epochs", type=int, default=300,
                        help="Number of epochs (default: 300)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="L2 regularization (default: 0)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-lnc-len", type=int, default=25000)
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (k-mer offset shifting)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit number of training chunks (default: all)")

    # Model hyperparameters
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--one-word", type=int, default=4)
    parser.add_argument("--ks", type=int, default=1)
    parser.add_argument("--no-sigmoid", action="store_true")
    parser.add_argument("--p0", type=float, default=0.5)

    # Validation frequency
    parser.add_argument("--val-every-n-chunks", type=int, default=50,
                        help="Validate after every N chunks (default: 50)")
    parser.add_argument("--save-every-n-epochs", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")

    # Output
    parser.add_argument("-o", "--outdir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "output_full"))
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="Device: -1 for CPU, 0+ for GPU")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

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
        print(f"Using CUDA device {args.device} - {torch.cuda.get_device_name(args.device)}")

    # Create output dir
    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "log.txt")

    # Save config
    config = vars(args).copy()
    config['device'] = str(config['device'])
    with open(os.path.join(args.outdir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Load sequence index
    seq_index = load_seq_index(args.seq_index)

    # Load validation datasets
    print("\nLoading validation datasets...")
    valid_datasets = {}
    for split_name in ["final_valid_unseen_pair", "final_valid_unseen_source",
                       "final_valid_unseen_target"]:
        path = os.path.join(args.valid_dir, f"{split_name}.jsonl")
        if os.path.exists(path):
            valid_datasets[split_name] = CiceklabDataset(path, seq_index, args.max_lnc_len)

    # Build k-mer dictionaries
    base_number_dict_mirna = {"A": 1, "G": 2, "C": 3, "U": 4}
    base_number_dict_lnc = build_kmer_dict(args.one_word)

    # Create or load model
    start_epoch = 0
    best_mcc = -1.0

    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint} ...")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model = ckpt['model']
        start_epoch = ckpt.get('epoch', 0)
        best_mcc = ckpt.get('best_mcc', -1.0)
        print(f"  Resuming from epoch {start_epoch}, best_mcc={best_mcc:.4f}")
    else:
        print("\nInitializing TEC-LncMir model...")
        contact_model = ContactCNN(args.ks, args.projection_dim)
        model = ModelInteraction(
            args, contact_model,
            do_sigmoid=not args.no_sigmoid, p0=args.p0,
        )

    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    header = (f"\n{'='*70}\n"
              f"  FULL TRAINING — TEC-LncMir on Ciceklab Data\n"
              f"  Epochs: {args.num_epochs}, Batch size: {args.batch_size}, LR: {args.lr}\n"
              f"  Augmentation: {'ON' if args.augment else 'OFF'}\n"
              f"  Max chunks: {args.max_chunks or 'ALL'}\n"
              f"  Validate every {args.val_every_n_chunks} chunks\n"
              f"{'='*70}\n")
    print(header)
    with open(log_path, 'a') as f:
        f.write(header)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()
        chunk_count = 0

        # Stream through training chunks
        for chunk_path, chunk_loader_raw in iter_training_chunks(
            args.train_chunks, seq_index,
            batch_size=args.batch_size,
            max_lnc_len=args.max_lnc_len,
            max_chunks=args.max_chunks,
            shuffle=True,
        ):
            chunk_name = os.path.basename(chunk_path)
            chunk_count += 1

            # If augmentation is on, wrap the dataset
            if args.augment:
                # We need to access the underlying dataset
                base_ds = chunk_loader_raw.dataset
                aug_ds = AugmentedChunkDataset(base_ds, k=args.one_word)
                loader = torch.utils.data.DataLoader(
                    aug_ds, batch_size=args.batch_size,
                    collate_fn=collate_paired_sequences,
                    shuffle=True, drop_last=False,
                )
            else:
                loader = chunk_loader_raw

            chunk_loss = 0
            chunk_total = 0

            for lnc_seqs, mi_seqs, y in loader:
                loss, preds, labels = train_step(
                    args, model, lnc_seqs, mi_seqs, y,
                    base_number_dict_mirna, base_number_dict_lnc, args.device,
                )
                optimizer.step()
                optimizer.zero_grad()

                bs = len(lnc_seqs)
                epoch_loss += loss * bs
                epoch_correct += sum(p == l for p, l in zip(preds, labels))
                epoch_total += bs
                chunk_loss += loss * bs
                chunk_total += bs

            # Log chunk progress
            if chunk_count % 10 == 0:
                avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
                avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
                elapsed = time.time() - epoch_start
                msg = (f"  [Epoch {epoch+1}] Chunk {chunk_count}: "
                       f"loss={avg_loss:.4f}, acc={avg_acc:.4f} ({elapsed:.0f}s)")
                print(msg)

            # Periodic validation
            if chunk_count % args.val_every_n_chunks == 0:
                print(f"\n  --- Validating after chunk {chunk_count} ---")
                for split_name, vd in valid_datasets.items():
                    metrics = evaluate(model, vd, base_number_dict_mirna,
                                     base_number_dict_lnc, args.device, args.batch_size)
                    print_metrics(split_name, metrics, log_path)

                    if 'unseen_pair' in split_name and metrics.get('mcc', -1) > best_mcc:
                        best_mcc = metrics.get('mcc', -1)
                        save_path = os.path.join(args.outdir, "best_model.sav")
                        model.cpu()
                        torch.save(model, save_path)
                        model.to(args.device)
                        print(f"\n  >>> New best model! MCC={best_mcc:.4f}")

                        with open(os.path.join(args.outdir, "best_metrics.txt"), 'w') as f:
                            f.write(f"Epoch: {epoch+1}, Chunk: {chunk_count}\n")
                            for k, v in metrics.items():
                                f.write(f"{k}: {v}\n")

                model.train()

        # End of epoch
        elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
        avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0

        msg = (f"\n  Epoch {epoch+1}/{args.num_epochs} complete: "
               f"loss={avg_loss:.4f}, acc={avg_acc:.4f}, "
               f"chunks={chunk_count}, samples={epoch_total:,}, "
               f"time={elapsed:.0f}s\n")
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg)

        # End-of-epoch validation
        print(f"  --- End-of-epoch validation ---")
        for split_name, vd in valid_datasets.items():
            metrics = evaluate(model, vd, base_number_dict_mirna,
                             base_number_dict_lnc, args.device, args.batch_size)
            print_metrics(split_name, metrics, log_path)

            if 'unseen_pair' in split_name and metrics.get('mcc', -1) > best_mcc:
                best_mcc = metrics.get('mcc', -1)
                save_path = os.path.join(args.outdir, "best_model.sav")
                model.cpu()
                torch.save(model, save_path)
                model.to(args.device)
                print(f"\n  >>> New best model! MCC={best_mcc:.4f}")

                with open(os.path.join(args.outdir, "best_metrics.txt"), 'w') as f:
                    f.write(f"Epoch: {epoch+1}\n")
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")

        model.train()

        # Save checkpoint periodically
        if (epoch + 1) % args.save_every_n_epochs == 0:
            ckpt_path = os.path.join(args.outdir, f"checkpoint_epoch{epoch+1}.pt")
            model.cpu()
            torch.save({
                'epoch': epoch + 1,
                'model': model,
                'best_mcc': best_mcc,
                'config': config,
            }, ckpt_path)
            model.to(args.device)
            print(f"  Checkpoint saved: {ckpt_path}")

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Best MCC: {best_mcc:.4f}")
    print(f"  Output: {args.outdir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
