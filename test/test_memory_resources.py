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
import json
import os
import tempfile
from contextlib import contextmanager
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
    common._RESIDUE_COUNT_CACHE.clear()
    with tempfile.TemporaryDirectory() as d:
        p = _write_fasta(d, "X", 137)
        assert common.residue_count(p) == 137
    # unreadable path degrades to 0 (dry-run safety)
    assert common.residue_count(os.path.join(d, "does_not_exist.fasta")) == 0


def test_residue_count_does_not_cache_missing_file():
    """Regression: an early read of a not-yet-created file must NOT cache 0, or
    length-aware sizing collapses to the base once the file later appears."""
    common._RESIDUE_COUNT_CACHE.clear()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "late.fasta")
        assert common.residue_count(p) == 0  # file absent now
        _write_fasta(d, "late", 321)         # produced by an upstream job later
        assert common.residue_count(p) == 321  # re-read, not the stale 0


def test_fold_total_tokens_sums_chains_and_copies():
    common._RESIDUE_COUNT_CACHE.clear()
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


def test_fold_total_tokens_af3_precomputed_feature_fallback():
    """When data/<chain>.fasta is absent (precomputed features), AF3 length comes
    from the <features_dir>/<chain>_af3_input.json fallback."""
    common._RESIDUE_COUNT_CACHE.clear()
    common._AF3_INPUT_COUNT_CACHE.clear()
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as feat:
        _write_fasta(d, "B", 300)  # only B has a FASTA; A is "precomputed"
        with open(os.path.join(feat, "A_af3_input.json"), "w") as fh:
            json.dump({"sequences": [{"protein": {"id": "A", "sequence": "M" * 250}}]}, fh)
        # without fallback A is unknown -> only B counted
        assert common.fold_total_tokens("A+B", d, "+") == 300
        # with AF3 fallback A is recovered from the json
        assert (
            common.fold_total_tokens("A+B", d, "+", features_dir=feat, is_af3=True) == 550
        )
        # fallback is AF3-only: AF2 precomputed stays at 0 for the missing chain
        assert common.fold_total_tokens("A+B", d, "+", features_dir=feat, is_af3=False) == 300


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


def test_af3_inference_defaults_cover_observed_gpu_demand_anchors():
    """AF3 host request (the unified-memory spill ceiling) must exceed the observed
    AF3 GPU-VRAM demand for the pairs documented in the AlphaJudge handoff."""
    d = common.INFERENCE_RAM_DEFAULTS["alphafold3"]
    kw = dict(base_mb=d["base_mb"], per_token_sq_mb=d["per_token_sq_mb"],
              scaling=1.1, safety=1.25, attempt=1)
    anchors = [(2066, 25), (4556, 82), (4836, 100)]  # (tokens, observed GB)
    for n, observed_gb in anchors:
        req_gb = common.estimate_inference_mem_mb(n, **kw) / 1000.0
        assert req_gb >= 1.2 * observed_gb, (n, req_gb, observed_gb)
        assert req_gb <= 2.7 * observed_gb, (n, req_gb, observed_gb)
    sizes = [common.estimate_inference_mem_mb(n, **kw) for n in (200, 1000, 2000, 4000)]
    assert sizes == sorted(sizes)


def test_af2_inference_defaults_cover_measured_host_rss():
    """AF2 host request must cover the measured AF2 inference host RSS (which IS the
    consumed memory for AF2), with margin."""
    d = common.INFERENCE_RAM_DEFAULTS["alphafold2"]
    kw = dict(base_mb=d["base_mb"], per_token_sq_mb=d["per_token_sq_mb"],
              scaling=1.1, safety=1.25, attempt=1)
    measured = [(1583, 16.9), (2256, 30.8), (2324, 30.8)]  # (tokens, measured host RSS GB)
    for n, rss_gb in measured:
        req_gb = common.estimate_inference_mem_mb(n, **kw) / 1000.0
        assert req_gb >= 1.2 * rss_gb, (n, req_gb, rss_gb)
        assert req_gb <= 3.2 * rss_gb, (n, req_gb, rss_gb)


def test_backend_defaults_af2_heavier_than_af3():
    assert common.normalize_backend("af3") == "alphafold3"
    assert common.normalize_backend("AlphaFold2") == "alphafold2"
    assert common.normalize_backend(None) == "alphafold2"
    f2, f3 = common.FEATURE_RAM_DEFAULTS["alphafold2"], common.FEATURE_RAM_DEFAULTS["alphafold3"]
    i2, i3 = common.INFERENCE_RAM_DEFAULTS["alphafold2"], common.INFERENCE_RAM_DEFAULTS["alphafold3"]
    assert f2["base_mb"] > f3["base_mb"]
    assert f2["per_residue_mb"] > f3["per_residue_mb"]
    assert i2["base_mb"] > i3["base_mb"]
    assert i2["per_token_sq_mb"] > i3["per_token_sq_mb"]


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


@contextmanager
def _preserve_container_bind_environment():
    names = ("APPTAINER_BINDPATH", "SINGULARITY_BINDPATH")
    previous = {name: os.environ[name] for name in names if name in os.environ}
    try:
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(previous)


def test_prepare_container_binds_accepts_exact_adapter_owned_paths():
    with (
        tempfile.TemporaryDirectory() as directory,
        _preserve_container_bind_environment(),
    ):
        os.environ.pop("APPTAINER_BINDPATH", None)
        os.environ.pop("SINGULARITY_BINDPATH", None)

        common.prepare_container_binds(
            output_directory=directory,
            config={},
            extra_paths=(Path("/external/mmseqs-databases"),),
        )

        binds = os.environ["APPTAINER_BINDPATH"].split(",")
        assert "/external/mmseqs-databases:/external/mmseqs-databases" in binds
        assert "/opt:/opt" not in binds


def test_prepare_container_binds_preserves_user_binds_and_adds_required_paths():
    with (
        tempfile.TemporaryDirectory() as directory,
        _preserve_container_bind_environment(),
    ):
        os.environ["APPTAINER_BINDPATH"] = "/custom:/custom"
        os.environ["SINGULARITY_BINDPATH"] = "/custom:/custom"

        common.prepare_container_binds(
            output_directory=directory,
            config={},
            extra_paths=(Path("/external/mmseqs-databases"),),
        )

        binds = os.environ["APPTAINER_BINDPATH"].split(",")
        assert "/custom:/custom" in binds
        assert "/external/mmseqs-databases:/external/mmseqs-databases" in binds
        assert "/opt:/opt" not in binds


def test_prepare_container_binds_includes_resolved_extra_directory():
    with (
        tempfile.TemporaryDirectory() as directory,
        _preserve_container_bind_environment(),
    ):
        os.environ.pop("APPTAINER_BINDPATH", None)
        os.environ.pop("SINGULARITY_BINDPATH", None)
        tmp_path = Path(directory)
        target = tmp_path / "real-mmseqs"
        target.mkdir()
        link = tmp_path / "mmseqs-link"
        link.symlink_to(target, target_is_directory=True)

        common.prepare_container_binds(
            output_directory=directory,
            config={},
            extra_paths=(link,),
        )

        binds = os.environ["APPTAINER_BINDPATH"].split(",")
        assert f"{link}:{link}" in binds
        assert f"{target}:{target}" in binds


# --- length filtering (issue #33 + total caps) -------------------------------
# Note: fold-spec parsing (name/copies/regions) is owned by `alphapulldown-input-parser`
# (>=0.5.0) and tested there; APS's `parse_fold_chains` is a thin (name, copies) adapter
# exercised end-to-end through `fold_total_tokens` and the filter logic below.


def test_fold_length_violation():
    chains = [("A", 300, 1), ("B", 2000, 1)]  # total 2300
    assert common.fold_length_violation(chains, 0, 0) is None  # limits off
    assert common.fold_length_violation(chains, 0, 5000) is None  # under total cap
    assert common.fold_length_violation(chains, 0, 1000) is not None  # over total cap
    assert common.fold_length_violation(chains, 1000, 0) is not None  # B over per-protein
    assert common.fold_length_violation(chains, 2500, 0) is None  # under per-protein
    # homo-dimer copies count toward the total
    assert common.fold_length_violation([("A", 600, 3)], 0, 1500) is not None
    # unknown length fails open (None treated as 0)
    assert common.fold_length_violation([("A", None, 1)], 0, 10) is None


def test_default_total_length_caps_af3_gt_af2():
    assert common.MAX_TOTAL_LENGTH_DEFAULTS["alphafold2"] == 5000
    assert common.MAX_TOTAL_LENGTH_DEFAULTS["alphafold3"] == 7000


def test_fetch_uniprot_length_parses_and_fails_open():
    import urllib.request as ur

    class _FakeResp:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    orig = ur.urlopen
    try:
        common.fetch_uniprot_length.cache_clear()
        ur.urlopen = lambda url, timeout=30.0: _FakeResp(b">sp|X\nMKLVMK\nAA\n")
        assert common.fetch_uniprot_length("FAKE_OK") == 8  # 6 + 2, header skipped

        def _boom(url, timeout=30.0):
            raise OSError("offline")

        ur.urlopen = _boom
        assert common.fetch_uniprot_length("FAKE_FAIL") == 0  # fail open, no crash
    finally:
        ur.urlopen = orig


def test_chain_residue_count_length_cache_fallback():
    """AF2 precomputed features: no FASTA and no AF3 JSON, but the parse-time
    length cache supplies the length."""
    common._RESIDUE_COUNT_CACHE.clear()
    with tempfile.TemporaryDirectory() as d:
        # no data/A.fasta exists; cache provides it
        assert common.chain_residue_count("A", d) == 0
        assert common.chain_residue_count("A", d, length_cache={"A": 412}) == 412


def test_af3_input_residue_count_skips_ligands():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.json")
        with open(p, "w") as fh:
            json.dump(
                {"sequences": [
                    {"protein": {"id": "A", "sequence": "M" * 100}},
                    {"ligand": {"id": "L", "ccdCodes": ["ATP"]}},  # no 'sequence'
                ]},
                fh,
            )
        assert common.af3_input_residue_count(p) == 100  # ligand contributes 0


def test_af3_input_residue_count_reads_compressed_features():
    """`--compress_features` writes `*_af3_input.json.xz`; callers pass the plain
    name, and resolving only that spelling returned 0 (no length-aware sizing)."""
    import lzma

    common._AF3_INPUT_COUNT_CACHE.clear()
    payload = {"sequences": [{"protein": {"id": "A", "sequence": "M" * 250}}]}
    with tempfile.TemporaryDirectory() as d:
        plain = os.path.join(d, "A_af3_input.json")
        with lzma.open(plain + ".xz", "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        # asked for the plain name, only the compressed file exists
        assert common.af3_input_residue_count(plain) == 250
        common._AF3_INPUT_COUNT_CACHE.clear()
        # and the compressed name works directly
        assert common.af3_input_residue_count(plain + ".xz") == 250
        common._AF3_INPUT_COUNT_CACHE.clear()
        # a genuinely absent feature is still 0, in either spelling
        assert common.af3_input_residue_count(os.path.join(d, "B_af3_input.json")) == 0


def test_required_gpu_vram_gb():
    # 0.0045 MB/token^2: N=4836 -> ~105 GB; headroom scales it
    assert round(common.required_gpu_vram_gb(4836, 0.0045)) == 105
    assert round(common.required_gpu_vram_gb(2066, 0.0045)) == 19
    assert common.required_gpu_vram_gb(4836, 0.0045, headroom=0.5) < 60


def test_gpu_exclude_nodes_vram_routing():
    tiers = [
        {"min_vram_gb": 24, "nodes": "n24a,n24b"},
        {"min_vram_gb": 48, "nodes": "n48a,n48b"},
        {"min_vram_gb": 80, "nodes": "n80a"},
    ]
    c = 0.0045
    # small complex (~3 GB) fits the smallest tier -> exclude nothing
    assert common.gpu_exclude_nodes(800, tiers, c) == ""
    # ~26 GB -> needs >=48 GB tier -> exclude the 24 GB nodes
    assert common.gpu_exclude_nodes(2400, tiers, c) == "n24a,n24b"
    # ~55 GB -> needs the 80 GB tier -> exclude 24 and 48 GB nodes
    assert common.gpu_exclude_nodes(3500, tiers, c) == "n24a,n24b,n48a,n48b"
    # bigger than every tier -> use largest tier (spill), exclude all smaller
    assert common.gpu_exclude_nodes(20000, tiers, c) == "n24a,n24b,n48a,n48b"
    # static extra excludes are always appended (fallback for old container/GPU incompatibilities)
    assert common.gpu_exclude_nodes(800, tiers, c, extra_exclude="gpu50,gpu51") == "gpu50,gpu51"
    assert (
        common.gpu_exclude_nodes(2400, tiers, c, extra_exclude="gpu50")
        == "n24a,n24b,gpu50"
    )
    # unsorted tiers handled; no tiers -> only the static excludes
    assert common.gpu_exclude_nodes(2400, [], c, extra_exclude="gpu50") == "gpu50"
    assert common.gpu_exclude_nodes(2400, [], c) == ""


def test_mem_mb_reaches_sbatch_via_real_plugin():
    """Integration: the value our model computes is what the SLURM plugin turns
    into `sbatch --mem`. Skips gracefully if the plugin isn't importable."""
    try:
        from snakemake_executor_plugin_slurm.submit_string import get_submit_command
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"  (skipped: plugin not importable: {exc})")
        return

    mem = common.estimate_inference_mem_mb(
        2300, base_mb=16000, per_token_sq_mb=0.0045, scaling=1.1, safety=1.25, attempt=1
    )

    class _Res(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

    class _Job:
        threads = 8
        resources = _Res(mem_mb=mem, runtime=600, qos="normal")

    params = {
        "run_uuid": "test",
        "slurm_logfile": "/tmp/test.log",
        "comment_str": "test",
        "account": "",
        "partition": "",
        "workdir": "",
    }
    try:
        cmd = get_submit_command(_Job(), params)
    except Exception as exc:  # pragma: no cover - plugin internals may change
        print(f"  (skipped: plugin API changed: {exc})")
        return
    assert f"--mem {mem}" in cmd, cmd


# ---------------------------------------------------------------------------
# AF3 JSON inputs (ligands etc.) — issue #41: a `*.json` token in a fold must be
# treated as a direct AF3 input, never as a protein to download / build features for.
# ---------------------------------------------------------------------------


def _write_af3_json(directory: str, name: str, *, protein_len: int = 0, ligand=None):
    """Write a minimal AF3 input JSON; optional protein sequence and/or ligand."""
    sequences = []
    if protein_len:
        sequences.append({"protein": {"id": "A", "sequence": "A" * protein_len}})
    if ligand:
        sequences.append({"ligand": {"id": "L", "ccdCodes": [ligand]}})
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        json.dump({"name": name, "sequences": sequences}, handle)
    return path


def test_is_json_input_detects_json_tokens():
    assert common.is_json_input("ligand.json")
    assert common.is_json_input("/path/to/LIGAND.JSON")  # case-insensitive
    assert not common.is_json_input("P12345")
    assert not common.is_json_input("Prot.fasta")


def test_split_fold_inputs_separates_proteins_and_json():
    # The reported case: protein + ligand JSON with a copy number.
    assert common.split_fold_inputs("P12345+ligand.json:80") == (
        ["P12345"],
        ["ligand.json"],
    )
    # Pure protein folds yield no JSON inputs; copies/regions are stripped.
    assert common.split_fold_inputs("P01258+P0AEZ3:2") == (["P01258", "P0AEZ3"], [])
    # Paths are reduced to a base (protein) / basename (json).
    assert common.split_fold_inputs("/p/Prot.fasta+/q/lig.json") == (
        ["Prot"],
        ["lig.json"],
    )
    # De-duplication, first-seen order preserved.
    assert common.split_fold_inputs("A+A+lig.json+lig.json") == (["A"], ["lig.json"])


def test_format_af3_requested_fold_passes_json_through():
    # Regression for #41: protein -> generated feature JSON; *.json left untouched.
    assert (
        common.format_af3_requested_fold("P12345+ligand.json:80")
        == "P12345_af3_input.json+ligand.json:80"
    )
    assert (
        common.format_af3_requested_fold("P01258+P0AEZ3:2")
        == "P01258_af3_input.json+P0AEZ3_af3_input.json:2"
    )
    assert common.format_af3_requested_fold("P01258:1-100:2") == (
        "P01258_af3_input.json:1-100:2"
    )


def test_chain_residue_count_reads_json_input():
    common._AF3_INPUT_COUNT_CACHE.clear()
    with tempfile.TemporaryDirectory() as d:
        # Ligand-only JSON has no polymer sequence -> contributes 0.
        _write_af3_json(d, "ligand.json", ligand="ATP")
        assert common.chain_residue_count("ligand.json", d, d, is_af3=True) == 0
        # A JSON carrying a protein sequence is counted by its polymer length.
        _write_af3_json(d, "complex.json", protein_len=150)
        assert common.chain_residue_count("complex.json", d, d, is_af3=True) == 150


def test_fold_total_tokens_counts_protein_not_ligand_json():
    common._RESIDUE_COUNT_CACHE.clear()
    common._AF3_INPUT_COUNT_CACHE.clear()
    with tempfile.TemporaryDirectory() as d:
        _write_fasta(d, "P12345", 200)
        _write_af3_json(d, "ligand.json", ligand="ATP")
        # Protein counted; ligand JSON adds 0 and does not error.
        assert (
            common.fold_total_tokens(
                "P12345+ligand.json:80", d, "+", features_dir=d, is_af3=True
            )
            == 200
        )


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
