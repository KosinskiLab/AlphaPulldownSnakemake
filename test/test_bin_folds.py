"""Unit tests for ``bin_folds`` (issue #48 small-job batching).

``common.smk`` is plain Python (stdlib-only imports), so it is loaded by path and
its functions tested directly. Run with ``python test/test_bin_folds.py`` or
``pytest test/test_bin_folds.py``.
"""

import importlib.machinery
import importlib.util
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[1] / "workflow" / "rules" / "common.smk"
_spec = importlib.util.spec_from_loader(
    "aps_common", importlib.machinery.SourceFileLoader("aps_common", str(_COMMON))
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
bin_folds = common.bin_folds


def test_batch_size_one_is_unbatched_and_preserves_order():
    folds = [("B+C", 800), ("A", 100), ("D", 400)]
    assert bin_folds(folds, batch_size=1) == [["B+C"], ["A"], ["D"]]
    # batch_size <= 0 behaves the same (disabled)
    assert bin_folds(folds, batch_size=0) == [["B+C"], ["A"], ["D"]]


def test_groups_by_size_up_to_batch_size():
    folds = [("big", 900), ("t1", 100), ("t2", 120), ("t3", 110), ("t4", 130)]
    batches = bin_folds(folds, batch_size=2)
    # sorted ascending, packed two-per-batch -> small folds cluster together
    assert batches == [["t1", "t3"], ["t2", "t4"], ["big"]]
    # every fold appears exactly once
    assert sorted(f for b in batches for f in b) == ["big", "t1", "t2", "t3", "t4"]


def test_max_batch_tokens_caps_summed_work():
    folds = [("a", 300), ("b", 300), ("c", 300), ("d", 300)]
    batches = bin_folds(folds, batch_size=10, max_batch_tokens=700)
    # 300+300 <= 700 but +300 > 700 -> two per batch
    assert batches == [["a", "b"], ["c", "d"]]


def test_af2_batches_do_not_mix_monomer_and_multimer_runners():
    folds = [
        ("complex_small+partner", 100),
        ("monomer_small", 110),
        ("complex_large+partner", 300),
        ("monomer_large", 200),
    ]

    batches = bin_folds(
        folds,
        batch_size=4,
        max_batch_tokens=350,
        backend="alphafold2",
        delimiter="+",
    )

    # Each resident batch has one runner configuration. Size order and the token
    # cap still apply independently inside those configuration groups.
    assert batches == [
        ["complex_small+partner"],
        ["complex_large+partner"],
        ["monomer_small", "monomer_large"],
    ]
    assert bin_folds(
        folds,
        batch_size=4,
        max_batch_tokens=350,
        backend="af2",
        delimiter="+",
    ) == batches


def test_af3_batching_remains_configuration_agnostic():
    folds = [("complex+partner", 100), ("monomer", 110)]

    assert bin_folds(
        folds,
        batch_size=2,
        backend="alphafold3",
        delimiter="+",
    ) == [["complex+partner", "monomer"]]


def test_single_oversized_fold_still_forms_a_batch():
    folds = [("huge", 5000), ("small", 50)]
    batches = bin_folds(folds, batch_size=5, max_batch_tokens=1000)
    # 'huge' alone exceeds the cap but must not be dropped
    assert ["huge"] in batches
    assert sorted(f for b in batches for f in b) == ["huge", "small"]


def test_empty_input():
    assert bin_folds([], batch_size=4) == []


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
