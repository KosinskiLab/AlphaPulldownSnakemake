"""Tests for the length-aware memory model and the linear_resources passthrough.

``workflow/rules/common.smk`` is pure Python, so it is loaded directly here. The
``linear_resources`` callables are exercised exactly the way Snakemake invokes
them (wildcards positional; ``input``/``attempt`` by keyword) to guard the
input-passthrough refactor.

Run standalone:  python test/test_memory_resources.py
Or with pytest:  pytest test/test_memory_resources.py
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import tempfile
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[1] / "workflow" / "rules" / "common.smk"
_loader = importlib.machinery.SourceFileLoader("aps_common", str(_COMMON))
_spec = importlib.util.spec_from_loader("aps_common", _loader)
common = importlib.util.module_from_spec(_spec)
_loader.exec_module(common)


def _write_fasta(directory: str, name: str, length: int) -> str:
    path = os.path.join(directory, f"{name}.fasta")
    with open(path, "w") as handle:
        handle.write(f">{name}\n")
        # split across lines to confirm multi-line sequences are summed
        seq = "A" * length
        for i in range(0, length, 60):
            handle.write(seq[i : i + 60] + "\n")
    return path


def test_residue_count_counts_sequence_only():
    common.residue_count.cache_clear()
    with tempfile.TemporaryDirectory() as d:
        p = _write_fasta(d, "X", 137)
        assert common.residue_count(p) == 137
    # unreadable path degrades to 0 (dry-run safety)
    assert common.residue_count(os.path.join(d, "does_not_exist.fasta")) == 0


def test_fold_total_tokens_sums_chains_and_copies():
    common.residue_count.cache_clear()
    with tempfile.TemporaryDirectory() as d:
        _write_fasta(d, "A", 200)
        _write_fasta(d, "B", 300)
        assert common.fold_total_tokens("A+B", d, "+") == 500
        # copy number: homo-dimer counts twice
        assert common.fold_total_tokens("A:2", d, "+") == 400
        # region selection is conservatively counted at full length
        assert common.fold_total_tokens("A:1-100", d, "+") == 200
        # mixed
        assert common.fold_total_tokens("A:2+B", d, "+") == 700


def test_feature_mem_model_math():
    # safety * (base + per_residue * L), attempt 1 has no extra escalation
    val = common.estimate_feature_mem_mb(
        500, base_mb=64000, per_residue_mb=30, scaling=1.1, safety=1.25, attempt=1
    )
    assert val == int(1.25 * (64000 + 30 * 500))  # 98750
    # retry escalation multiplies by scaling ** (attempt - 1)
    val2 = common.estimate_feature_mem_mb(
        500, base_mb=64000, per_residue_mb=30, scaling=1.1, safety=1.25, attempt=3
    )
    assert val2 == int(1.25 * (64000 + 30 * 500) * (1.1 ** 2))


def test_inference_mem_model_math_and_cap():
    val = common.estimate_inference_mem_mb(
        1000, base_mb=24000, per_token_sq_mb=0.0045, scaling=1.1, safety=1.25, attempt=1
    )
    assert val == int(1.25 * (24000 + 0.0045 * 1000 ** 2))  # 35625
    # cap is honoured
    capped = common.estimate_inference_mem_mb(
        5000, base_mb=24000, per_token_sq_mb=0.0045, scaling=1.1, safety=1.25,
        attempt=1, cap_mb=50000,
    )
    assert capped == 50000


def test_inference_model_covers_observed_oom_anchors_with_margin():
    """Requested RAM must exceed the empirically observed AF3 peak demand for the
    pairs documented in the AlphaJudge handoff, with a sane (not wasteful) margin."""
    kw = dict(base_mb=24000, per_token_sq_mb=0.0045, scaling=1.1, safety=1.25, attempt=1)
    anchors = [  # (total_tokens, observed_peak_GB)
        (2066, 25),   # O00194+Q9ULV0
        (4556, 82),   # P02549+P11277
        (4836, 100),  # Q01082+Q13813
    ]
    for n, observed_gb in anchors:
        req_gb = common.estimate_inference_mem_mb(n, **kw) / 1000.0
        assert req_gb >= 1.2 * observed_gb, (n, req_gb, observed_gb)
        assert req_gb <= 2.5 * observed_gb, (n, req_gb, observed_gb)
    # monotonic in size
    sizes = [common.estimate_inference_mem_mb(n, **kw) for n in (200, 1000, 2000, 4000)]
    assert sizes == sorted(sizes)


def test_linear_resources_forwards_input_to_new_style_callbacks():
    # Snakemake invokes the resource callable as f(wildcards, input=..., attempt=...)
    res = common.linear_resources(
        mem_fn=lambda wildcards, input, attempt: 1000 * len(input) + attempt
    )
    assert res["mem_mb"]({}, input=["a", "b", "c"], attempt=1) == 3001
    assert res["avg_mem"]({}, input=["a", "b", "c"], attempt=1) == int(3001 * 0.75)


def test_linear_resources_still_supports_legacy_callbacks():
    res = common.linear_resources(mem_fn=lambda wc, attempt: 5000 * attempt)
    assert res["mem_mb"]({}, input=[], attempt=2) == 10000


def test_linear_resources_default_scaling_without_callbacks():
    res = common.linear_resources(mem=800, runtime=10)
    assert res["mem_mb"]({}, input=[], attempt=3) == 2400
    assert res["runtime"]({}, input=[], attempt=2) == 20
    assert res["attempt"]({}, input=[], attempt=4) == 4


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
