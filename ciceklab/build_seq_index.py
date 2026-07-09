#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_seq_index.py
==================
One-time preprocessing script that scans rna.fa and builds a fast
pickle lookup dictionary mapping RNA IDs → sequences.

Only keeps records whose IDs match known miRNA / lncRNA patterns:
  - hsa-miR-*, hsa-mir-*, MI0*, MIMAT*          (miRNA)
  - NONHSAG*, NONHSAT*, NONMMUG*, NONMMUT*      (NONCODE lncRNA)
  - ENST*, ENSMUSG*, ENSG*                       (Ensembl transcripts)

Usage:
    python build_seq_index.py [--rna-fa PATH] [--output PATH]

Outputs:
    seq_index.pkl  — dict[str, str] mapping ID → uppercase RNA sequence
"""

import argparse
import os
import pickle
import re
import sys
import time


# Patterns that identify relevant RNA records
KEEP_PATTERNS = [
    re.compile(r'^hsa-mi[Rr]', re.IGNORECASE),   # mature miRNA names
    re.compile(r'^MI\d{5,}'),                       # miRBase precursor IDs
    re.compile(r'^MIMAT\d+'),                       # miRBase mature IDs
    re.compile(r'^NONHSAG\d+'),                     # NONCODE human gene
    re.compile(r'^NONHSAT\d+'),                      # NONCODE human transcript
    re.compile(r'^NONMMUG\d+'),                      # NONCODE mouse gene
    re.compile(r'^NONMMUT\d+'),                      # NONCODE mouse transcript
    re.compile(r'^ENST\d+'),                         # Ensembl transcript
    re.compile(r'^ENSMUSG\d+'),                      # Ensembl mouse gene
    re.compile(r'^ENSG\d+'),                         # Ensembl human gene
]


def should_keep(record_id: str) -> bool:
    """Check if a FASTA record ID matches any of our keep patterns."""
    for pat in KEEP_PATTERNS:
        if pat.match(record_id):
            return True
    return False


def build_index_streaming(fasta_path: str) -> dict:
    """
    Build the sequence index by streaming the FASTA file line-by-line.
    This avoids loading the entire 1.8GB file into memory via BioPython.
    """
    seq_index = {}
    current_id = None
    current_seq_parts = []
    records_scanned = 0
    records_kept = 0

    print(f"Scanning {fasta_path} ...")
    start_time = time.time()

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Save previous record if it matched
                if current_id is not None and should_keep(current_id):
                    seq = ''.join(current_seq_parts).upper()
                    # Normalize T → U for RNA
                    seq = seq.replace('T', 'U')
                    seq_index[current_id] = seq
                    records_kept += 1

                records_scanned += 1
                if records_scanned % 2_000_000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Scanned {records_scanned:,} records, "
                          f"kept {records_kept:,} ... ({elapsed:.0f}s)")

                # Parse the new record ID (first whitespace-delimited token)
                header = line[1:]  # remove '>'
                current_id = header.split()[0].split('|')[0]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Don't forget the last record
    if current_id is not None and should_keep(current_id):
        seq = ''.join(current_seq_parts).upper()
        seq = seq.replace('T', 'U')
        seq_index[current_id] = seq
        records_kept += 1
    records_scanned += 1

    elapsed = time.time() - start_time
    print(f"\nDone! Scanned {records_scanned:,} records in {elapsed:.1f}s")
    print(f"Kept {records_kept:,} sequences in index")

    return seq_index


def print_summary(seq_index: dict):
    """Print a summary of what's in the index."""
    mirna_count = 0
    lncrna_noncode = 0
    lncrna_ensembl = 0
    other = 0

    lengths = []
    for rid, seq in seq_index.items():
        lengths.append(len(seq))
        if rid.startswith('hsa-mi') or rid.startswith('MI0') or rid.startswith('MIMAT'):
            mirna_count += 1
        elif rid.startswith('NON'):
            lncrna_noncode += 1
        elif rid.startswith('ENS'):
            lncrna_ensembl += 1
        else:
            other += 1

    print(f"\nIndex summary:")
    print(f"  miRNA records:          {mirna_count:,}")
    print(f"  lncRNA (NONCODE):       {lncrna_noncode:,}")
    print(f"  lncRNA/RNA (Ensembl):   {lncrna_ensembl:,}")
    print(f"  Other:                  {other:,}")
    print(f"  Total:                  {len(seq_index):,}")
    if lengths:
        print(f"  Sequence lengths: min={min(lengths)}, "
              f"max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")


def main():
    parser = argparse.ArgumentParser(description="Build sequence index from rna.fa")
    parser.add_argument("--rna-fa", type=str,
                        default=os.path.join(os.path.dirname(__file__), "rna.fa"),
                        help="Path to rna.fa (default: ciceklab/rna.fa)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__), "seq_index.pkl"),
                        help="Output pickle file path (default: ciceklab/seq_index.pkl)")
    args = parser.parse_args()

    if not os.path.exists(args.rna_fa):
        print(f"ERROR: {args.rna_fa} not found!")
        sys.exit(1)

    seq_index = build_index_streaming(args.rna_fa)
    print_summary(seq_index)

    print(f"\nSaving to {args.output} ...")
    with open(args.output, 'wb') as f:
        pickle.dump(seq_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved! File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
