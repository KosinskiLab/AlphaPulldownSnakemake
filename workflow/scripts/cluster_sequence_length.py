#!python3
import argparse

from alphapulldown.utils.modelling_setup import (
    parse_fold,
    create_custom_info,
    create_interactors,
)
from alphapulldown.objects import MultimericObject


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster folds by sequence length."
    )
    parser.add_argument(
        "--folds",
        dest="folds",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="All the jobs to fold",
    )
    parser.add_argument(
        "--protein_delimiter",
        dest="protein_delimiter",
        type=str,
        default="+",
        required=False,
        help="protein list files",
    )
    parser.add_argument(
        "--features_directory",
        dest="features_directory",
        type=str,
        nargs="+",
        required=False,
        help="Path to computed monomer features.",
    )
    parser.add_argument(
        "--bin_size",
        dest="bin_size",
        type=int,
        required=False,
        default=150,
        help="Bin size used for clustering sequences.",
    )
    parser.add_argument(
        "--output_file",
        dest="output_file",
        type=str,
        default="sequence_clusters.txt",
        required=False,
        help="Path to comma separated output file.",
    )
    args = parser.parse_args()

    try:
        args.folds = snakemake.params.folds
        args.output_file = snakemake.output[0]
        args.protein_delimiter = snakemake.params.protein_delimiter
        args.bin_size = snakemake.params.cluster_bin_size
        args.features_directory = [snakemake.params.feature_directory, ]
    except Exception:
        pass

    if args.folds is None:
        raise ValueError("--folds needs to be specified.")

    return args


def main():
    args = parse_args()

    all_jobs = {"name": [], "msa_depth": [], "seq_length": []}
    for idx, i in enumerate(args.folds):
        parsed_input = parse_fold(
            [i], args.features_directory, args.protein_delimiter
        )

        data = create_custom_info(parsed_input)
        interactors = create_interactors(data, args.features_directory, 0)
        multimer = MultimericObject(interactors[0])

        msa_depth, seq_length = multimer.feature_dict["msa"].shape
        all_jobs["name"].append(i)
        all_jobs["msa_depth"].append(msa_depth)
        all_jobs["seq_length"].append(seq_length)

    # Assign elements to bins
    min_seq_length = max(min(all_jobs["seq_length"]), 1)
    all_jobs["cluster"] = [
        int((value - min_seq_length) // args.bin_size) for value in all_jobs["seq_length"]
    ]
    label_stats = {}
    for index, label in enumerate(all_jobs["cluster"]):
        if label not in label_stats:
            label_stats[label] = {"max_seq_length" : 0, "max_msa_depth" : 0}
        label_stats[label]["max_seq_length"] = max(
            label_stats[label]["max_seq_length"], all_jobs["seq_length"][index]
        )
        label_stats[label]["max_msa_depth"] = max(
            label_stats[label]["max_msa_depth"], all_jobs["msa_depth"][index]
        )

    all_jobs["max_msa_depth"], all_jobs["max_seq_length"] = [], []
    for label in all_jobs["cluster"]:
        all_jobs["max_msa_depth"].append(label_stats[label]["max_msa_depth"])
        all_jobs["max_seq_length"].append(label_stats[label]["max_seq_length"])

    with open(args.output_file, mode = "w", encoding = "utf-8") as ofile:
        ofile.write(','.join([str(x) for x in list(all_jobs.keys())]) + "\n")
        for fold in zip(*all_jobs.values()):
            _ = ofile.write(','.join([str(x) for x in fold]) + "\n")

if __name__ == "__main__":
    main()
