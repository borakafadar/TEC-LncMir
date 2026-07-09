#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_seq_index.py
==================
One-time preprocessing script that:
  1. Scans all training chunks and validation/test JSONL files to collect
     the exact set of unique miRNA and lncRNA IDs used in the dataset.
  2. Scans rna.fa in a single streaming pass to extract sequences for
     those exact IDs.
  3. Saves the resulting dictionary to seq_index.pkl.

Usage:
    python build_seq_index.py [--rna-fa PATH] [--output PATH]

Outputs:
    seq_index.pkl — dict[str, str] mapping ID → uppercase RNA sequence
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time


def collect_needed_ids(train_chunks_dir: str, valid_test_dir: str) -> set:
    """
    Collect all unique miRNA and lncRNA IDs referenced across
    training chunks and validation/test datasets.
    """
    needed_ids = set()
    total_records = 0
    mirna_lncrna_pairs = 0

    # Collect from training chunks
    chunk_files = sorted(glob.glob(os.path.join(train_chunks_dir, "chunk_*.jsonl")))
    print(f"Collecting unique IDs from {len(chunk_files)} training chunks...")
    for cf in chunk_files:
        with open(cf, 'r') as f:
            for line in f:
                total_records += 1
                record = json.loads(line.strip())
                if record.get('interaction_type') != 'rna-rna':
                    continue
                t1 = record.get('RNA_type', '').lower()
                t2 = record.get('target_RNA_type', '').lower()
                if (t1 == 'mirna' and t2 == 'lncrna') or (t1 == 'lncrna' and t2 == 'mirna'):
                    mirna_lncrna_pairs += 1
                    needed_ids.add(record['RNA_id'])
                    needed_ids.add(record['target_id'])

    # Collect from validation/test datasets
    eval_files = sorted(glob.glob(os.path.join(valid_test_dir, "*.jsonl")))
    print(f"Collecting unique IDs from {len(eval_files)} evaluation files...")
    for ef in eval_files:
        with open(ef, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get('interaction_type') != 'rna-rna':
                    continue
                t1 = record.get('RNA_type', '').lower()
                t2 = record.get('target_RNA_type', '').lower()
                if (t1 == 'mirna' and t2 == 'lncrna') or (t1 == 'lncrna' and t2 == 'mirna'):
                    needed_ids.add(record['RNA_id'])
                    needed_ids.add(record['target_id'])

    print(f"  Found {mirna_lncrna_pairs:,} miRNA-lncRNA pairs across datasets.")
    print(f"  Total unique required IDs to extract: {len(needed_ids):,}")
    return needed_ids


def build_index_streaming(fasta_path: str, needed_ids: set) -> dict:
    """
    Build the sequence index by streaming rna.fa line-by-line,
    keeping only IDs present in needed_ids.
    """
    seq_index = {}
    current_id = None
    current_seq_parts = []
    records_scanned = 0

    print(f"\nScanning {fasta_path} for {len(needed_ids):,} exact IDs...")
    start_time = time.time()

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Save previous record if it matched
                if current_id is not None and current_id in needed_ids:
                    seq = ''.join(current_seq_parts).upper().replace('T', 'U')
                    seq_index[current_id] = seq

                records_scanned += 1
                if records_scanned % 2_000_000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Scanned {records_scanned:,} records, "
                          f"found {len(seq_index):,}/{len(needed_ids):,} IDs... ({elapsed:.0f}s)")

                # Parse FASTA header ID (first token before whitespace or pipe)
                header = line[1:]
                current_id = header.split()[0].split('|')[0]
                current_seq_parts = []
            else:
                if current_id in needed_ids:
                    current_seq_parts.append(line)

    # Last record
    if current_id is not None and current_id in needed_ids:
        seq = ''.join(current_seq_parts).upper().replace('T', 'U')
        seq_index[current_id] = seq
    records_scanned += 1

    elapsed = time.time() - start_time
    print(f"\nDone! Scanned {records_scanned:,} FASTA records in {elapsed:.1f}s")
    print(f"Successfully matched and indexed {len(seq_index):,} / {len(needed_ids):,} required IDs.")

    missing_ids = needed_ids - set(seq_index.keys())
    if missing_ids:
        print(f"Note: {len(missing_ids)} IDs referenced in datasets were not found in {os.path.basename(fasta_path)}.")

    return seq_index


def main():
    parser = argparse.ArgumentParser(description="Build exact sequence index from rna.fa based on dataset IDs")
    parser.add_argument("--rna-fa", type=str,
                        default=os.path.join(os.path.dirname(__file__), "rna.fa"),
                        help="Path to rna.fa")
    parser.add_argument("--train-chunks", type=str,
                        default=os.path.join(os.path.dirname(__file__), "training_chunks"),
                        help="Path to training_chunks directory")
    parser.add_argument("--valid-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "data_with_negatives", "rna_rna", "miRNA_lncRNA"),
                        help="Path to evaluation datasets directory")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__), "seq_index.pkl"),
                        help="Output pickle file path")
    args = parser.parse_args()

    if not os.path.exists(args.rna_fa):
        print(f"ERROR: {args.rna_fa} not found!")
        sys.exit(1)

    needed_ids = collect_needed_ids(args.train_chunks, args.valid_dir)
    seq_index = build_index_streaming(args.rna_fa, needed_ids)

    print(f"\nSaving to {args.output} ...")
    with open(args.output, 'wb') as f:
        pickle.dump(seq_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved! File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
