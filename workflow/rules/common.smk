""" Snakemake I/O and utility functions

    Copyright (c) 2024 European Molecular Biology Laboratory

    Authors: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""


def feature_suffix(compression : str = "lzma") -> str:
    _compression = {
        "lzma" : "xz",
    }
    suffix = _compression.get(compression, None)
    ret = "pkl"
    if suffix is not None:
        ret += f".{suffix}"
    return ret
