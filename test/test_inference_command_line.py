"""The inference flags in the config reach the structure_inference command line.

Unit tests cover ``batch_inference_args`` and the flag tables; these dry runs check the
last step, that what they produce is what the job would actually run. They use the
self-contained resident-batch fixtures (two proteins, precomputed features) with the
output directory moved to ``tmp_path``, and skip when Snakemake is not installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURES = _REPOSITORY / "test" / "fixtures" / "resident_batch"
_BACKENDS = {"alphafold2": "af2.yaml", "alphafold3": "af3.yaml"}
# Set in the rule's shell so the shared compile cache is safe on network filesystems.
_XLA_CACHES_EXPORT = (
    'export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES='
    '"${JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES:-none}"'
)
# One AlphaPulldown invocation: the script at the start of a line, then its
# backslash-continued arguments. "command -v run_structure_prediction_batch.py" is not one.
_INVOCATION = re.compile(
    r"^[ \t]*(run_structure_prediction(?:_batch)?\.py)[ \t]*\\\n((?:[^\n]*\\\n)*[^\n]*)",
    re.MULTILINE,
)


def _write_config(tmp_path, backend, *, batch_size, inference_arguments):
    """The backend's fixture config, with ``inference_arguments`` written verbatim as the
    structure_inference_arguments block, so YAML reads it exactly as a user's file."""
    config = yaml.safe_load((_FIXTURES / _BACKENDS[backend]).read_text(encoding="utf-8"))
    config["output_directory"] = str(tmp_path / "output")
    config["batch_size"] = batch_size
    del config["structure_inference_arguments"]
    text = (
        yaml.safe_dump(config, sort_keys=False)
        + "structure_inference_arguments:\n"
        + textwrap.indent(f"--fold_backend: {backend}\n{inference_arguments}", "  ")
    )
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path, yaml.safe_load(text)


def _dry_run(config_path):
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        pytest.skip("Snakemake executable is not available")
    completed = subprocess.run(
        [
            snakemake,
            "--snakefile", str(_REPOSITORY / "workflow" / "Snakefile"),
            "--configfile", str(config_path),
            "--cores", "1",
            "--rerun-triggers", "mtime",
            "--dry-run",
            "--printshellcmds",
        ],
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "APPTAINER_BINDPATH": "", "SINGULARITY_BINDPATH": ""},
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    return completed


def _structure_inference_jobs(stdout):
    """The printed structure_inference jobs, each from its header to the next job's."""
    jobs = re.split(r"^(?=(?:local)?rule \w+:$)", stdout, flags=re.MULTILINE)
    return [job for job in jobs if job.startswith("rule structure_inference:")]


def _invocations(job):
    """(script, {flag: value}) for each AlphaPulldown call in one job's shell command."""
    return [
        (script, dict(re.findall(r"--(\w+)=(\S+)", arguments)))
        for script, arguments in _INVOCATION.findall(job)
    ]


@pytest.mark.parametrize("backend", sorted(_BACKENDS))
def test_fast_kernels_and_cache_reach_every_inference_call(tmp_path, backend):
    """A resident batch: both the batch command and the per-fold fallback get the flags."""
    cache = tmp_path / "compile-cache"
    config_path, config = _write_config(
        tmp_path,
        backend,
        batch_size=2,
        # Unquoted, as users write it: YAML 1.1 reads a bare on as the boolean True.
        inference_arguments=f"--fast_kernels: on\n--jax_compilation_cache_dir: {cache}\n",
    )
    assert config["structure_inference_arguments"]["--fast_kernels"] is True

    completed = _dry_run(config_path)
    log = completed.stdout + completed.stderr

    # The parse-time guard accepts --fast_kernels for both backends.
    assert "[inference-flags]" not in log
    jobs = _structure_inference_jobs(completed.stdout)
    assert len(jobs) == 1, "two folds with batch_size 2 make one resident batch"
    job = jobs[0]
    calls = _invocations(job)
    assert sorted(script for script, _flags in calls) == [
        "run_structure_prediction.py", "run_structure_prediction_batch.py",
    ]
    for script, flags in calls:
        # The boolean is passed as Python prints it. AlphaPulldown's --fast_kernels is a
        # string flag that reads true/false (any case) as on/off, so True means on.
        assert flags["fast_kernels"] == "True", script
        assert flags["jax_compilation_cache_dir"] == str(cache), script
        assert flags["fold_backend"] == backend, script
        # --allow_resume is added to AlphaFold 2 batches only; AlphaFold 3 rejects it.
        assert ("allow_resume" in flags) == (backend == "alphafold2"), script
    assert job.index(_XLA_CACHES_EXPORT) < job.index("run_structure_prediction")


@pytest.mark.parametrize("backend", sorted(_BACKENDS))
def test_default_cache_reaches_each_single_fold_job(tmp_path, backend):
    """batch_size 1: one job per fold, each pointed at the workflow's default cache."""
    config_path, config = _write_config(
        tmp_path, backend, batch_size=1, inference_arguments="--fast_kernels: \"auto\"\n"
    )

    completed = _dry_run(config_path)

    default_cache = os.path.join(config["output_directory"], ".jax_compilation_cache")
    jobs = _structure_inference_jobs(completed.stdout)
    assert len(jobs) == 2, "one job per fold"
    for job in jobs:
        calls = _invocations(job)
        assert calls, job
        for script, flags in calls:
            assert flags["fast_kernels"] == "auto", script
            assert flags["jax_compilation_cache_dir"] == default_cache, script
            assert "allow_resume" not in flags, script
        assert job.index(_XLA_CACHES_EXPORT) < job.index("run_structure_prediction")
