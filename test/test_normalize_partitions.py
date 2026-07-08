"""Unit tests for ``normalize_partitions`` (multi-partition support).

``common.smk`` is plain Python (stdlib-only imports), so it is loaded by path and
its functions tested directly. Run with ``python test/test_normalize_partitions.py``
or ``pytest test/test_normalize_partitions.py``.
"""

import importlib.machinery
import importlib.util
import shlex
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[1] / "workflow" / "rules" / "common.smk"
_spec = importlib.util.spec_from_loader(
    "aps_common", importlib.machinery.SourceFileLoader("aps_common", str(_COMMON))
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
normalize_partitions = common.normalize_partitions


def test_single_partition_unchanged():
    assert normalize_partitions("gpu-el8") == "gpu-el8"


def test_none_and_empty_return_none():
    assert normalize_partitions(None) is None
    assert normalize_partitions("") is None
    assert normalize_partitions("   ") is None
    assert normalize_partitions([]) is None
    assert normalize_partitions([" ", ""]) is None


def test_comma_separated_string_is_normalized():
    assert normalize_partitions("gpu-el8,transform") == "gpu-el8,transform"
    # surrounding whitespace around commas is stripped
    assert normalize_partitions("gpu-el8, transform ,training") == (
        "gpu-el8,transform,training"
    )


def test_whitespace_separated_string():
    assert normalize_partitions("gpu-el8 transform") == "gpu-el8,transform"


def test_yaml_list_is_joined_with_commas():
    assert normalize_partitions(["gpu-el8", "transform"]) == "gpu-el8,transform"
    assert normalize_partitions(("gpu-el8", "gpu-training")) == "gpu-el8,gpu-training"


def test_duplicates_removed_order_preserved():
    assert normalize_partitions(["gpu-el8", "transform", "gpu-el8"]) == (
        "gpu-el8,transform"
    )
    assert normalize_partitions("transform,gpu-el8,transform") == (
        "transform,gpu-el8"
    )


def test_list_entries_are_stripped():
    assert normalize_partitions([" gpu-el8 ", " transform"]) == "gpu-el8,transform"


def test_result_survives_shlex_quote_unquoted():
    # The SLURM plugin runs the partition through shlex.quote before passing it to
    # `sbatch -p`. A comma-joined list must survive that untouched (comma is a
    # shell-safe character) so sbatch receives the full partition list.
    result = normalize_partitions("gpu-el8,transform")
    assert shlex.quote(result) == result == "gpu-el8,transform"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all normalize_partitions tests passed")
