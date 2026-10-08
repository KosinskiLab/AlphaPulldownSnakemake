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
_XLA_VARIABLES = (
    "XLA_CLIENT_MEM_FRACTION",
    "TF_FORCE_UNIFIED_MEMORY",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
)
# A MIG node: --query-gpu reports the parent card, -L lists the slice with its UUID.
_MIG_NVIDIA_SMI = """\
if [ "$1" = "-L" ]; then
    echo "GPU 0: NVIDIA RTX PRO 4500 Blackwell (UUID: GPU-0)"
    echo "  MIG 1g.16gb     Device  0: (UUID: {uuid})"
else
    echo 32623
fi
"""
# The inference base RAM that "auto" keeps for the process itself.
_BASE_RAM_MB = {"alphafold2": 16000, "alphafold3": 8000}


def _unified_memory_block(tmp_path, backend="alphafold3", **extra):
    config_path, _config = _write_config(
        tmp_path, backend, batch_size=2, inference_arguments="", **extra
    )
    jobs = _structure_inference_jobs(_dry_run(config_path).stdout)
    assert len(jobs) == 1, "two folds with batch_size 2 make one resident batch"
    match = _UNIFIED_MEMORY_BLOCK.search(jobs[0])
    assert match, jobs[0]
    return match.group(0)


def _run_unified_memory_block(tmp_path, block, nvidia_smi, host_mb, cuda_visible_devices=None):
    """Run the rendered block as the job does, in bash strict mode, with ``nvidia_smi``
    (a shell script body) as nvidia-smi and a stale XLA_PYTHON_CLIENT_MEM_FRACTION.

    A dry run prints the job's host RAM as a number or, when it is resolved only once the
    job runs, as <TBD> (depending on the Snakemake version); ``host_mb`` replaces it.
    Returns the [unified-memory] log fields (None when the block logged nothing) and the
    XLA variables as the block left them.
    """
    block, replaced = re.subn(r"(-v r=|host_mem_mb=)(?:<TBD>|\d+)", rf"\g<1>{host_mb}", block)
    assert replaced == 2, block
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "nvidia-smi"
    fake.write_text("#!/bin/sh\n" + nvidia_smi, encoding="utf-8")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    for name in ("CUDA_VISIBLE_DEVICES", *_XLA_VARIABLES):
        env.pop(name, None)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    report = "".join(f'\necho "env {name}=${{{name}-<unset>}}"' for name in _XLA_VARIABLES)
    script = "set -euo pipefail\n" + block + report + "\n"
    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    variables = dict(re.findall(r"^env (\w+)=(.*)$", completed.stdout, re.MULTILINE))
    logged = _UNIFIED_MEMORY_LOG.search(completed.stdout)
    if logged is None:
        return None, variables
    assert int(logged.group(1)) == host_mb
    margin_mb, gpu_mb, fraction = logged.group(2, 3, 4)
    fields = {"margin_mb": int(margin_mb), "gpu_mb": int(gpu_mb) if gpu_mb else None}
    return {**fields, "fraction": fraction}, variables


def _assert_unified_memory_exported(variables, fraction):
    assert variables == {
        "XLA_CLIENT_MEM_FRACTION": fraction,
        "TF_FORCE_UNIFIED_MEMORY": "true",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        # With both names set JAX's CUDA plugin fails and it falls back to CPU, so the
        # deprecated one is cleared.
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "<unset>",
    }


@pytest.mark.parametrize("backend", sorted(_BACKENDS))
def test_auto_fraction_lets_the_job_spill_into_its_host_ram(tmp_path, backend):
    """XLA's limit is fraction x VRAM and holds VRAM and host spill together, so "auto" is
    1 + (host RAM - base inference RAM) / VRAM, never below 1. Host RAM / VRAM alone
    capped the whole job at its host RAM and left about one GPU's worth of it unused."""
    block = _unified_memory_block(tmp_path, backend)
    margin_mb = _BASE_RAM_MB[backend]

    # 128 GiB on an 80 GB H100: VRAM plus 120 GiB (AF3) or 112 GiB (AF2) of host spill,
    # where host RAM / VRAM gave 1.607.
    logged, variables = _run_unified_memory_block(tmp_path, block, "echo 81559\n", 131072)
    expected = {"alphafold3": "2.509", "alphafold2": "2.411"}[backend]
    assert logged == {"margin_mb": margin_mb, "gpu_mb": 81559, "fraction": expected}
    _assert_unified_memory_exported(variables, expected)

    for nvidia_smi, cuda_visible_devices, host_mb, gpu_mb in (
        # Several GPUs listed: the first one's VRAM.
        ("printf '81559\\n46068\\n'\n", None, 131072, 81559),
        # A MIG slice is sized by its own profile, not the parent card's 32623 MiB...
        (_MIG_NVIDIA_SMI.format(uuid="MIG-slice"), "MIG-slice", 32000, 16384),
        # ...and falls back to --query-gpu, rather than failing, when it is not listed.
        (_MIG_NVIDIA_SMI.format(uuid="MIG-other"), "MIG-slice", 32000, 32623),
        # Host RAM at or below the margin still gets the whole GPU.
        ("echo 81559\n", None, margin_mb, 81559),
        ("echo 81559\n", None, margin_mb // 2, 81559),
    ):
        logged, variables = _run_unified_memory_block(
            tmp_path, block, nvidia_smi, host_mb, cuda_visible_devices
        )
        fraction = f"{max(1.0, 1 + (host_mb - margin_mb) / gpu_mb):.3f}"
        assert logged == {"margin_mb": margin_mb, "gpu_mb": gpu_mb, "fraction": fraction}
        _assert_unified_memory_exported(variables, fraction)

    # No usable VRAM reading, on a whole card or a MIG slice: the fixed fallback.
    for nvidia_smi, cuda_visible_devices in (
        ("exit 1\n", None),
        ("echo '[N/A]'\n", None),
        ("exit 1\n", "MIG-slice"),
    ):
        logged, variables = _run_unified_memory_block(
            tmp_path, block, nvidia_smi, 131072, cuda_visible_devices
        )
        assert logged == {"margin_mb": margin_mb, "gpu_mb": None, "fraction": "3.2"}
        _assert_unified_memory_exported(variables, "3.2")


def test_auto_fraction_margin_follows_structure_inference_ram_bytes(tmp_path):
    block = _unified_memory_block(tmp_path, structure_inference_ram_bytes=20000)

    logged, variables = _run_unified_memory_block(tmp_path, block, "echo 81559\n", 16000)

    assert logged == {"margin_mb": 20000, "gpu_mb": 81559, "fraction": "1.000"}
    _assert_unified_memory_exported(variables, "1.000")


def test_pinned_fraction_is_exported_as_written(tmp_path):
    block = _unified_memory_block(tmp_path, structure_inference_xla_mem_fraction=2.4)

    logged, variables = _run_unified_memory_block(tmp_path, block, "echo 81559\n", 131072)

    assert logged is None, "a pinned fraction skips the auto computation"
    _assert_unified_memory_exported(variables, "2.4")


def test_unified_memory_off_leaves_the_environment_alone(tmp_path):
    block = _unified_memory_block(tmp_path, structure_inference_unified_memory=False)

    logged, variables = _run_unified_memory_block(tmp_path, block, "echo 81559\n", 131072)

    assert logged is None
    assert variables == {
        "XLA_CLIENT_MEM_FRACTION": "<unset>",
        "TF_FORCE_UNIFIED_MEMORY": "<unset>",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "<unset>",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
    }
