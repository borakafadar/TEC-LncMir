#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_ciceklab.py
====================
Post-training evaluation script for TEC-LncMir on ciceklab test data.

Evaluates a trained model on all 3 test splits:
  - final_test_unseen_pair.jsonl      (unseen pair combinations)
  - final_test_unseen_source.jsonl    (unseen miRNAs)
  - final_test_unseen_target.jsonl    (unseen lncRNAs)

Reports: Accuracy, Sensitivity, Specificity, PPV, NPV, F1, MCC, AUROC, AUPR

Usage:
    python evaluate_ciceklab.py --model output_full/best_model.sav --device 0
    python evaluate_ciceklab.py --model output_smoke/best_model.sav --device -1
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from code.utils import get_tokens, get_tokens_word

from data_adapter import (
    load_seq_index,
    CiceklabDataset,
    collate_paired_sequences,
)


# ---------------------------------------------------------------------------
# Tokenization
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, dataset, base_number_dict_mirna, base_number_dict_lnc,
                   device, batch_size=16, save_predictions_path=None):
    """
    Evaluate model on a dataset.
    Returns (metrics_dict, predictions_array, labels_array).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=collate_paired_sequences, shuffle=False,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for lnc_seqs, mi_seqs, y in tqdm(loader, desc="Evaluating"):
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

    # Save raw predictions
    if save_predictions_path:
        with open(save_predictions_path, 'w') as f:
            f.write("label\tprediction\n")
            for l, p in zip(labels, preds):
                f.write(f"{l}\t{p:.6f}\n")

    # Compute metrics
    if len(set(labels)) < 2:
        pred_binary = (preds >= 0.5).astype(int)
        return {"accuracy": accuracy_score(labels, pred_binary), "n_samples": len(labels)}, preds, labels

    pred_binary = (preds >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
    denom = ((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))

    metrics = {
        "n_samples": int(len(labels)),
        "positive_samples": int(tp + fn),
        "negative_samples": int(tn + fp),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        "ppv": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
        "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0,
        "mcc": float((tp * tn - fp * fn) / (denom ** 0.5)) if denom > 0 else 0,
        "auroc": float(roc_auc_score(labels, preds)),
        "aupr": float(average_precision_score(labels, preds)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return metrics, preds, labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(labels, preds, split_name, outdir):
    """Generate and save evaluation plots."""
    prefix = os.path.join(outdir, f"{split_name}_")

    # 1. Prediction distribution
    pos_phat = preds[labels == 1]
    neg_phat = preds[labels == 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Prediction Distribution — {split_name}")
    ax1.hist(pos_phat, bins=50, color='green', alpha=0.7)
    ax1.set_xlim(0, 1)
    ax1.set_title("Positive pairs")
    ax1.set_xlabel("Predicted probability")
    ax2.hist(neg_phat, bins=50, color='red', alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.set_title("Negative pairs")
    ax2.set_xlabel("Predicted probability")
    plt.tight_layout()
    plt.savefig(prefix + "pred_dist.svg", dpi=150)
    plt.savefig(prefix + "pred_dist.png", dpi=150)
    plt.close()

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(labels, preds)
    auroc = roc_auc_score(labels, preds)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve — {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + "roc.svg", dpi=150)
    plt.savefig(prefix + "roc.png", dpi=150)
    plt.close()

    # 3. Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    aupr = average_precision_score(labels, preds)
    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, color='blue', lw=2, where='post',
             label=f'AUPR = {aupr:.4f}')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='blue')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + "pr.svg", dpi=150)
    plt.savefig(prefix + "pr.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate TEC-LncMir on Ciceklab Test Data")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.sav file)")
    parser.add_argument("--seq-index", type=str,
                        default=os.path.join(os.path.dirname(__file__), "seq_index.pkl"))
    parser.add_argument("--test-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "data_with_negatives", "rna_rna", "miRNA_lncRNA"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-lnc-len", type=int, default=25000)
    parser.add_argument("-o", "--outdir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "output_eval"))
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="Device: -1 for CPU, 0+ for GPU")
    args = parser.parse_args()

    # Device
    if args.device == -1:
        device = 'cpu'
        print("Using CPU")
    else:
        device = args.device
        print(f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}")

    # Create output dir
    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.model} ...")
    model = torch.load(args.model, map_location='cpu')
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} parameters")
    print(f"  k-mer size: {model.one_word}")

    # Load sequence index
    seq_index = load_seq_index(args.seq_index)

    # Build k-mer dictionaries
    base_number_dict_mirna = {"A": 1, "G": 2, "C": 3, "U": 4}
    base_number_dict_lnc = build_kmer_dict(model.one_word)

    # Test splits to evaluate
    test_splits = [
        "final_test_unseen_pair",
        "final_test_unseen_source",
        "final_test_unseen_target",
    ]

    all_results = {}

    print(f"\n{'='*70}")
    print(f"  EVALUATION — TEC-LncMir on Ciceklab Test Data")
    print(f"{'='*70}")

    for split_name in test_splits:
        path = os.path.join(args.test_dir, f"{split_name}.jsonl")
        if not os.path.exists(path):
            print(f"\n  WARNING: {path} not found, skipping")
            continue

        print(f"\n  Loading {split_name} ...")
        dataset = CiceklabDataset(path, seq_index, args.max_lnc_len)

        if len(dataset) == 0:
            print(f"  WARNING: No valid pairs found in {split_name}")
            continue

        pred_path = os.path.join(args.outdir, f"{split_name}_predictions.tsv")
        metrics, preds, labels = evaluate_model(
            model, dataset, base_number_dict_mirna, base_number_dict_lnc,
            device, args.batch_size, save_predictions_path=pred_path,
        )

        all_results[split_name] = metrics

        # Print metrics
        print(f"\n  === {split_name} ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:>18s}: {v:.4f}")
            else:
                print(f"    {k:>18s}: {v}")

        # Generate plots
        if len(set(labels)) >= 2:
            plot_results(labels, preds, split_name, args.outdir)
            print(f"  Plots saved to {args.outdir}/{split_name}_*.png")

    # Save combined results
    results_path = os.path.join(args.outdir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  All results saved to {results_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Split':<30s} {'Acc':>7s} {'F1':>7s} {'MCC':>7s} {'AUROC':>7s} {'AUPR':>7s}")
    print(f"  {'-'*65}")
    for split_name, metrics in all_results.items():
        short = split_name.replace('final_test_', '')
        print(f"  {short:<30s} "
              f"{metrics.get('accuracy', 0):>7.4f} "
              f"{metrics.get('f1', 0):>7.4f} "
              f"{metrics.get('mcc', 0):>7.4f} "
              f"{metrics.get('auroc', 0):>7.4f} "
              f"{metrics.get('aupr', 0):>7.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
