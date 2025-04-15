#!python3
""" Utility to split multi-line fasta into individual files for use in pulldowns.

    Copyright (c) 2025 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>

    Example:

    python3 split_fasta.py /path/file.fasta /path/output --output-list samplesheet.csv

"""
import os
import gzip
import argparse
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

UNWANTED_SYMBOLS = ["|", "=", "&", "*", "@", "#", "`", ":", ";", "$", "?"]

@dataclass
class FastaEntry:
    identifier: str  # UniProt ID
    description: str  # Full header line
    sequence: str  # Sequence content

def is_gzipped(filepath: str) -> bool:
    """Check if a file is gzipped by looking at its magic bytes."""
    with open(filepath, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

def parse_fasta(fasta_file: str) -> Iterator[FastaEntry]:
    """Parse a FASTA file and yield entries one by one."""
    opener, mode = open, "r"
    if is_gzipped(fasta_file):
        opener, mode = gzip.open, "rt"

    with opener(fasta_file, mode) as f:
        current_id = None
        current_description = None
        current_sequence = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id:
                    yield FastaEntry(
                        identifier=current_id,
                        description=current_description,
                        sequence="".join(current_sequence),
                    )

                # Start a new entry
                # Assuming UniProt format: >sp|P12345|GENE_HUMAN Description
                # or >tr|A0A123|GENE_HUMAN Description
                header_parts = line[1:].split("|")
                if len(header_parts) >= 2:
                    current_id = header_parts[1]
                else:
                    # Fallback if not in expected format
                    current_id = line[1:].split()[0]
                    for symbol in UNWANTED_SYMBOLS:
                        if symbol in current_id:
                            current_id = current_id.replace(symbol, "_")
                    print(
                        f"'{line}' is not in UniProt format."
                        f" Using '{current_id}' as sequence id."
                    )

                current_description = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)

        if current_id:
            yield FastaEntry(
                identifier=current_id,
                description=current_description,
                sequence="".join(current_sequence),
            )


def write_fasta_entry(entry: FastaEntry, output_dir: str) -> str:
    filename = f"{entry.identifier}.fasta"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w") as f:
        f.write(f">{entry.description}\n")

        for i in range(0, len(entry.sequence), 60):
            f.write(f"{entry.sequence[i:i+60]}\n")

    return file_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a multi-sequence FASTA file into individual files"
    )
    parser.add_argument("fasta_file", help="Input multi-sequence FASTA file")
    parser.add_argument(
        "output_dir", help="Output directory for individual FASTA files"
    )
    parser.add_argument(
        "--output-list", help="Output file for the list of created file paths"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Number of threads to use"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []

        for entry in parse_fasta(args.fasta_file):
            future = executor.submit(write_fasta_entry, entry, str(output_dir))
            futures.append(future)

        for future in futures:
            file_path = future.result()
            file_paths.append(file_path)

    if args.output_list:
        with open(args.output_list, "w") as f:
            for path in file_paths:
                f.write(f"{path}\n")
        print(f"List of file paths written to {args.output_list}")

    print(
        f"Split {len(file_paths)} sequences from {args.fasta_file} into {args.output_dir}"
    )


if __name__ == "__main__":
    main()
