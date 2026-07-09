#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_adapter.py
===============
PyTorch Dataset adapters that bridge ciceklab JSONL data format
to the interface expected by TEC-LncMir's training code.

Classes:
    CiceklabDataset         — For validation/test JSONL files (loaded fully into memory)
    CiceklabChunkDataset    — For streaming training chunks one-at-a-time
    CiceklabMergedDataset   — For loading all lncRNA-miRNA pairs from chunks into memory

All datasets return (lncRNA_sequence, miRNA_sequence, label) tuples,
matching the interface of the paper's collate_paired_sequences.
"""

import json
import os
import pickle
import glob
import random
import torch
import torch.utils.data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_mirna_lncrna_pair(record: dict) -> tuple:
    """
    Check if a JSONL record is a miRNA-lncRNA interaction pair.
    Returns (mirna_id, lncrna_id, label) or None.

    Handles BOTH orderings:
      - RNA_type=miRNA, target_RNA_type=lncRNA  →  (RNA_id, target_id)
      - RNA_type=lncRNA, target_RNA_type=miRNA  →  (target_id, RNA_id)
    """
    if record.get('interaction_type') != 'rna-rna':
        return None

    rna_type = record.get('RNA_type', '').lower()
    target_rna_type = record.get('target_RNA_type', '').lower()

    if rna_type == 'mirna' and target_rna_type == 'lncrna':
        return (record['RNA_id'], record['target_id'], record['interaction_label'])
    elif rna_type == 'lncrna' and target_rna_type == 'mirna':
        # Vice-versa: lncRNA is RNA_id, miRNA is target_id
        return (record['target_id'], record['RNA_id'], record['interaction_label'])
    else:
        return None


def load_seq_index(index_path: str) -> dict:
    """Load the prebuilt sequence index pickle."""
    print(f"Loading sequence index from {index_path} ...")
    with open(index_path, 'rb') as f:
        seq_index = pickle.load(f)
    print(f"  Loaded {len(seq_index):,} sequences")
    return seq_index


def resolve_sequence(seq_id: str, seq_index: dict, max_len: int = 25000) -> str:
    """
    Look up a sequence by ID, returning None if not found or too long.
    Normalizes to uppercase RNA (T→U).
    """
    seq = seq_index.get(seq_id)
    if seq is None:
        return None
    if len(seq) > max_len:
        return None
    # Ensure uppercase and RNA bases
    seq = seq.upper().replace('T', 'U')
    return seq


def extract_pairs_from_jsonl(filepath: str, seq_index: dict,
                              max_lnc_len: int = 25000) -> list:
    """
    Extract all miRNA-lncRNA pairs from a JSONL file.
    Returns list of (lncrna_seq, mirna_seq, label).
    Skips pairs where either ID is not found in the sequence index.
    """
    pairs = []
    skipped_no_seq = 0
    skipped_too_long = 0
    total_mirna_lncrna = 0

    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            result = is_mirna_lncrna_pair(record)
            if result is None:
                continue

            mirna_id, lncrna_id, label = result
            total_mirna_lncrna += 1

            mirna_seq = resolve_sequence(mirna_id, seq_index, max_len=500)
            lncrna_seq = resolve_sequence(lncrna_id, seq_index, max_len=max_lnc_len)

            if mirna_seq is None or lncrna_seq is None:
                if mirna_seq is None:
                    skipped_no_seq += 1
                if lncrna_seq is None:
                    if lncrna_id in seq_index and len(seq_index[lncrna_id]) > max_lnc_len:
                        skipped_too_long += 1
                    else:
                        skipped_no_seq += 1
                continue

            pairs.append((lncrna_seq, mirna_seq, label))

    return pairs, total_mirna_lncrna, skipped_no_seq, skipped_too_long


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class CiceklabDataset(torch.utils.data.Dataset):
    """
    Dataset for validation/test JSONL files.
    Loads all pairs into memory (these files are small, ~500-14K records).

    Returns (lncrna_sequence, mirna_sequence, label) tuples.
    The sequences are raw strings — tokenization happens in the training loop
    just like the original TEC-LncMir code.
    """

    def __init__(self, jsonl_path: str, seq_index: dict, max_lnc_len: int = 25000):
        self.pairs = []
        pairs, total, skip_seq, skip_long = extract_pairs_from_jsonl(
            jsonl_path, seq_index, max_lnc_len
        )
        self.pairs = pairs
        name = os.path.basename(jsonl_path)
        print(f"  [{name}] Loaded {len(self.pairs)} pairs "
              f"(from {total} miRNA-lncRNA records, "
              f"skipped: {skip_seq} no-seq, {skip_long} too-long)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        lncrna_seq, mirna_seq, label = self.pairs[i]
        return lncrna_seq, mirna_seq, label


class CiceklabChunkDataset(torch.utils.data.Dataset):
    """
    Dataset for a single training chunk JSONL file.
    Filters for miRNA-lncRNA pairs and resolves sequences.
    """

    def __init__(self, chunk_path: str, seq_index: dict, max_lnc_len: int = 25000):
        pairs, total, skip_seq, skip_long = extract_pairs_from_jsonl(
            chunk_path, seq_index, max_lnc_len
        )
        self.pairs = pairs
        self.chunk_name = os.path.basename(chunk_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        lncrna_seq, mirna_seq, label = self.pairs[i]
        return lncrna_seq, mirna_seq, label


class CiceklabMergedDataset(torch.utils.data.Dataset):
    """
    Dataset that loads ALL miRNA-lncRNA pairs from all training chunks
    into memory. Use this when you have enough RAM (~5-10GB) and want
    full random shuffling across all chunks.

    Alternative to streaming chunks one-at-a-time.
    """

    def __init__(self, chunks_dir: str, seq_index: dict,
                 max_lnc_len: int = 25000, max_chunks: int = None):
        self.pairs = []
        chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.jsonl")))
        if max_chunks is not None:
            chunk_files = chunk_files[:max_chunks]

        total_pairs = 0
        total_records = 0
        total_skip_seq = 0
        total_skip_long = 0

        for i, chunk_path in enumerate(chunk_files):
            pairs, total, skip_seq, skip_long = extract_pairs_from_jsonl(
                chunk_path, seq_index, max_lnc_len
            )
            self.pairs.extend(pairs)
            total_pairs += len(pairs)
            total_records += total
            total_skip_seq += skip_seq
            total_skip_long += skip_long

            if (i + 1) % 50 == 0 or (i + 1) == len(chunk_files):
                print(f"  Loaded {i + 1}/{len(chunk_files)} chunks, "
                      f"{total_pairs:,} pairs so far ...")

        print(f"\nMerged dataset: {len(self.pairs):,} pairs total "
              f"(from {total_records:,} miRNA-lncRNA records, "
              f"skipped: {total_skip_seq:,} no-seq, {total_skip_long:,} too-long)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        lncrna_seq, mirna_seq, label = self.pairs[i]
        return lncrna_seq, mirna_seq, label


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_paired_sequences(args):
    """
    Collate function for PyTorch DataLoader.
    Matches the interface of the original TEC-LncMir code.

    Input:  list of (lncrna_seq, mirna_seq, label) tuples
    Output: (lncrna_seqs, mirna_seqs, labels_tensor)
    """
    x0 = [a[0] for a in args]  # lncRNA sequences
    x1 = [a[1] for a in args]  # miRNA sequences
    y = [a[2] for a in args]   # labels
    return x0, x1, torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Chunk iterator for streaming training
# ---------------------------------------------------------------------------

def iter_training_chunks(chunks_dir: str, seq_index: dict,
                          batch_size: int, max_lnc_len: int = 25000,
                          max_chunks: int = None, shuffle: bool = True):
    """
    Generator that yields DataLoaders, one per training chunk.
    This is the streaming approach: process one chunk at a time.

    Usage:
        for chunk_loader in iter_training_chunks(chunks_dir, seq_index, batch_size=16):
            for (lnc_seqs, mi_seqs, labels) in chunk_loader:
                # training step
    """
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.jsonl")))
    if max_chunks is not None:
        chunk_files = chunk_files[:max_chunks]

    # Shuffle chunk order each epoch for variety
    if shuffle:
        random.shuffle(chunk_files)

    for chunk_path in chunk_files:
        dataset = CiceklabChunkDataset(chunk_path, seq_index, max_lnc_len)
        if len(dataset) == 0:
            continue
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_paired_sequences,
            shuffle=shuffle,
            drop_last=False,
        )
        yield chunk_path, loader
