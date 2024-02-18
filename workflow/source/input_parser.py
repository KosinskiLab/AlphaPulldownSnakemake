from typing import Tuple, List, Set, Dict


class InputParser:
    def __init__(
        self,
        fold_specifications: Tuple[str],
        unique_sequences: Tuple[str],
        sequences_by_fold: Dict[str, Set],
    ):
        self.fold_specifications = fold_specifications
        self.unique_sequences = unique_sequences
        self.sequences_by_fold = sequences_by_fold

    @staticmethod
    def _parse_alphaabriss_format(
        fold_specifications: List[str],
    ) -> Tuple[Tuple, Dict[str, Set]]:
        unique_sequences, sequences_by_fold = set(), {}

        for fold_specification in fold_specifications:
            sequences = set()
            for fold in fold_specification.split(";"):
                sequences.add(fold.split(":")[0])

            unique_sequences.update(sequences)
            sequences_by_fold[fold_specification] = sequences

        return tuple(unique_sequences), sequences_by_fold

    @classmethod
    def from_file(cls, filepath: str, file_format: str = "alphaabriss"):
        with open(filepath, mode="r") as infile:
            data = tuple(set([line.strip() for line in infile.readlines()]))

        match file_format:
            case "alphaabriss":
                unique_sequences, sequences_by_fold = cls._parse_alphaabriss_format(
                    data
                )

            case _:
                raise ValueError(f"Format {file_format} is not supported.")

        return cls(
            fold_specifications=data,
            unique_sequences=unique_sequences,
            sequences_by_fold=sequences_by_fold,
        )
