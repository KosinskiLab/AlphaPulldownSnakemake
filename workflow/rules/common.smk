""" Snakemake I/O and utility functions

    Copyright (c) 2024 European Molecular Biology Laboratory

    Authors: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import itertools
from contextlib import nullcontext
from os import makedirs, symlink
from os.path import join, splitext, basename, exists
from typing import Tuple, Dict, List, Set, Union, TextIO


def read_file(filepath : str):
    with open(filepath, mode = "r", encoding = "utf-8") as file:
        lines = file.read().splitlines()
    return list(line.lstrip().rstrip() for line in lines if line)

def process_files(input_files : List[str],
                  output_path : Union[str, TextIO] = None,
                  delimiter : str = '+'):
    """Process the input files to compute the Cartesian product and write to the output file."""
    lists_of_lines = [read_file(filepath) for filepath in input_files]
    cartesian_product = list(itertools.product(*lists_of_lines))
    if output_path is None:
        return itertools.product(*lists_of_lines)
    else:
        context_manager = nullcontext(output_path)
        if isinstance(output_path, str):
            context_manager = open(output_path, mode = "w", encoding = "utf-8")

        with context_manager as output_file:
            for combination in cartesian_product:
                output_file.write(delimiter.join(combination) + '\n')


class InputParser:
    def __init__(
        self,
        fold_specifications: Tuple[str],
        sequences_by_origin: Dict[str, List[str]],
        sequences_by_fold: Dict[str, Set],
    ):
        self.fold_specifications = fold_specifications
        self.sequences_by_origin = sequences_by_origin
        self.sequences_by_fold = sequences_by_fold

        unique_sequences = set()
        for value in self.sequences_by_origin.values():
            unique_sequences.update(
                set([splitext(basename(x))[0] for x in value])
            )
        self.unique_sequences = unique_sequences

    @staticmethod
    def _strip_path_and_extension(filepath : str) -> str:
        return splitext(basename(filepath))[0]

    @staticmethod
    def _parse_alphaabriss_format(
        fold_specifications: List[str],
        protein_delimiter : str = "_"
    ) -> Tuple[Dict[str, List[str]], Dict[str, Set]]:
        unique_sequences, sequences_by_fold = set(), {}

        for fold_specification in fold_specifications:
            sequences = set()
            clean_fold_specification = []
            for fold in fold_specification.split(protein_delimiter):
                fold = fold.split(":")
                sequences.add(fold[0])

                protein_name = splitext(basename(fold[0]))[0]
                clean_fold_specification.append(":".join([protein_name, *fold[1:]]))

            clean_fold_specification = protein_delimiter.join([str(x) for x in clean_fold_specification])

            unique_sequences.update(sequences)
            sequences_by_fold[clean_fold_specification] = {splitext(basename(x))[0] for x in sequences}

        sequences_by_origin = {
            "uniprot" : [],
            "local" : []
        }
        for sequence in unique_sequences:
            if not exists(sequence):
                sequences_by_origin["uniprot"].append(sequence)
                continue
            sequences_by_origin["local"].append(sequence)

        return sequences_by_origin, sequences_by_fold

    def symlink_local_files(self, output_directory : str) -> None:
        makedirs(output_directory, exist_ok = True)
        for file in self.sequences_by_origin["local"]:
            symlink(file, join(output_directory, basename(file)))
        return None

    @classmethod
    def from_file(cls, filepath: str, file_format: str = "alphaabriss", protein_delimiter : str = "_"):
        with open(filepath, mode="r") as infile:
            data = [line.strip() for line in infile.readlines() if len(line.strip())]
            data = tuple(set(data))

        match file_format:
            case "alphaabriss":
                ret = cls._parse_alphaabriss_format(
                    fold_specifications = data, protein_delimiter=protein_delimiter
                )
                sequences_by_origin, sequences_by_fold = ret

            case _:
                raise ValueError(f"Format {file_format} is not supported.")

        fold_specifications = list(sequences_by_fold.keys())
        return cls(
            fold_specifications=fold_specifications,
            sequences_by_origin=sequences_by_origin,
            sequences_by_fold=sequences_by_fold,
        )

    def update_clustering(self, data : Dict[str, List]) -> None:
        folds_by_cluster = {}
        for fold, cluster in zip(data["name"], data["cluster"]):
            if cluster not in folds_by_cluster:
                folds_by_cluster[cluster] = []
            folds_by_cluster[cluster].append(fold)

        sequences_by_fold, new_folds = {}, []
        for cluster, folds in folds_by_cluster.items():
            new_fold = " ".join([str(x) for x in folds])
            total_sequences = []
            for fold in folds:
                total_sequences.extend(self.sequences_by_fold[fold])
            sequences_by_fold[new_fold] = list(set(total_sequences))
            new_folds.append(new_fold)

        self.sequences_by_fold.update(sequences_by_fold)
        self.fold_specifications = new_folds

