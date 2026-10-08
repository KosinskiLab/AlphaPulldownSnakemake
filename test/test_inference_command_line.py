"""The inference flags in the config reach the structure_inference command line.

Unit tests cover ``batch_inference_args`` and the flag tables; these dry runs check the
last step, that what they produce is what the job would actually run. They use the
self-contained resident-batch fixtures (two proteins, precomputed features) with the
output directory moved to ``tmp_path``, and skip when Snakemake is not installed. The
last tests run the rendered unified-memory block against a fake nvidia-smi.
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


def _write_config(tmp_path, backend, *, batch_size, inference_arguments, **extra):
    """The backend's fixture config, with ``inference_arguments`` written verbatim as the
    structure_inference_arguments block, so YAML reads it exactly as a user's file.
    ``extra`` sets further top-level keys."""
    config = yaml.safe_load((_FIXTURES / _BACKENDS[backend]).read_text(encoding="utf-8"))
    config["output_directory"] = str(tmp_path / "output")
    config["batch_size"] = batch_size
    config.update(extra)
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


# The rendered unified-memory block: from its leading comment up to the inference call.
_UNIFIED_MEMORY_BLOCK = re.compile(
    r'^[ \t]*# "auto" is resolved here.*?'
    r'(?=^[ \t]*if \[ "(?:true|false)" = "true" \] && command -v)',
    re.MULTILINE | re.DOTALL,
)
_UNIFIED_MEMORY_LOG = re.compile(
    r"\[unified-memory\] host_mem_mb=(\d+) margin_mb=(\d+) gpu_mem_mb=(\d*) "
    r"-> XLA_CLIENT_MEM_FRACTION=(\S+)"
)
# A MIG slice: --query-gpu reports the parent card, -L the slice's own profile.
_MIG_NVIDIA_SMI = """\
if [ "$1" = "-L" ]; then
    echo "GPU 0: NVIDIA RTX PRO 4500 Blackwell (UUID: GPU-0)"
    echo "  MIG 1g.16gb     Device  0: (UUID: MIG-test-slice)"
else
    echo 32623
fi
"""


def _unified_memory_block(tmp_path, **extra):
    config_path, _config = _write_config(
        tmp_path, "alphafold3", batch_size=2, inference_arguments="", **extra
    )
    jobs = _structure_inference_jobs(_dry_run(config_path).stdout)
    assert len(jobs) == 1, "two folds with batch_size 2 make one resident batch"
    match = _UNIFIED_MEMORY_BLOCK.search(jobs[0])
    assert match, jobs[0]
    return match.group(0)


def _run_unified_memory_block(tmp_path, block, nvidia_smi, host_mb, cuda_visible_devices=None):
    """Run the rendered block with ``nvidia_smi`` (a shell script body) as nvidia-smi.

    A dry run prints the job's host RAM as <TBD> (it is resolved when the job runs), so
    ``host_mb`` stands in for it. Returns the logged margin and GPU VRAM (MB; None when
    not read), the logged fraction, and the XLA_CLIENT_MEM_FRACTION the block exported.
    """
    assert "host_mem_mb=<TBD>" in block
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "nvidia-smi"
    fake.write_text("#!/bin/sh\n" + nvidia_smi, encoding="utf-8")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    script = block.replace("<TBD>", str(host_mb)) + '\necho "exported=$XLA_CLIENT_MEM_FRACTION"\n'
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    logged = _UNIFIED_MEMORY_LOG.search(completed.stdout)
    assert logged, completed.stdout + completed.stderr
    assert int(logged.group(1)) == host_mb
    margin_mb, gpu_mb, fraction = logged.group(2, 3, 4)
    exported = re.search(r"^exported=(\S*)$", completed.stdout, re.MULTILINE).group(1)
    return int(margin_mb), int(gpu_mb) if gpu_mb else None, fraction, exported


def test_auto_fraction_lets_the_job_spill_into_its_host_ram(tmp_path):
    """XLA's limit is fraction x VRAM and holds VRAM and host spill together, so "auto" is
    1 + (host RAM - base inference RAM) / VRAM. Host RAM / VRAM alone capped the whole
    job at its host RAM and left about one GPU's worth of that RAM unused."""
    block = _unified_memory_block(tmp_path)

    # 128 GiB on an 80 GB H100: ~200 GiB in all, ~120 GiB of it host spill (R/V gave 1.607).
    margin_mb, gpu_mb, fraction, exported = _run_unified_memory_block(
        tmp_path, block, "echo 81559\n", host_mb=131072
    )
    assert (margin_mb, gpu_mb) == (8000, 81559), "the AlphaFold 3 inference base RAM"
    assert fraction == exported == "2.509"

    # A 16 GB MIG slice is sized by its own profile, not the parent card's 32623 MiB.
    margin_mb, gpu_mb, fraction, exported = _run_unified_memory_block(
        tmp_path, block, _MIG_NVIDIA_SMI, host_mb=32000, cuda_visible_devices="MIG-test-slice"
    )
    assert gpu_mb == 16384
    assert fraction == exported == f"{1 + (32000 - margin_mb) / 16384:.3f}"

    # Without a readable GPU the fixed fallback applies.
    _margin_mb, gpu_mb, fraction, exported = _run_unified_memory_block(
        tmp_path, block, "exit 1\n", host_mb=131072
    )
    assert gpu_mb is None
    assert fraction == exported == "3.2"


def test_auto_fraction_margin_follows_base_ram_and_floors_at_one(tmp_path):
    """The margin is structure_inference_ram_bytes. A host request below it still gets
    the whole GPU: the fraction floors at 1, as XLA itself does."""
    block = _unified_memory_block(tmp_path, structure_inference_ram_bytes=20000)

    margin_mb, _gpu_mb, fraction, exported = _run_unified_memory_block(
        tmp_path, block, "echo 81559\n", host_mb=16000
    )

    assert margin_mb == 20000
    assert fraction == exported == "1.000"
