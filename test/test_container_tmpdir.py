"""Container shells repair an inherited TMPDIR before native tools write into it.

Like the inference command-line tests, these render the real workflow with the
resident-batch fixtures and execute its shell prologue under bash strict mode.
No container image, GPU, database, or cluster allocation is needed.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURES = _REPOSITORY / "test" / "fixtures" / "resident_batch"
_COMMON = _REPOSITORY / "workflow" / "rules" / "common.smk"
_loader = importlib.machinery.SourceFileLoader("aps_common", str(_COMMON))
_spec = importlib.util.spec_from_loader("aps_common", _loader)
common = importlib.util.module_from_spec(_spec)
_loader.exec_module(common)


def _dry_run(tmp_path, **extra):
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        pytest.skip("Snakemake executable is not available")
    config = yaml.safe_load((_FIXTURES / "af3.yaml").read_text(encoding="utf-8"))
    config.update(
        output_directory=str(tmp_path / "output"),
        structure_inference_unified_memory=False,
        **extra,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
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
    jobs = re.split(r"^(?=(?:local)?rule \w+:$)", completed.stdout, flags=re.MULTILINE)
    return [job for job in jobs if re.match(r"(?:local)?rule \w+:$", job, re.MULTILINE)]


def test_inference_native_tempfile_write_recovers_from_unavailable_tmpdir(tmp_path):
    """A native writer uses TMPDIR directly, unlike Python's tempfile fallback."""
    fallback = tmp_path / "container-tmp"
    jobs = _dry_run(tmp_path, container_tmpdir=str(fallback))
    job, = [job for job in jobs if job.startswith("rule structure_inference:")]
    prologue = re.split(
        r'if \[ "(?:true|false)" = "true" \] && command -v run_structure_prediction_batch.py',
        job.split("Shell command:", 1)[1],
        maxsplit=1,
    )[0]
    # An inaccessible parent makes the inherited path unusable even when tests
    # run as root; it stands in for a job-local directory absent in the container.
    blocked = tmp_path / "unbound-scratch"
    blocked.write_text("not a mounted directory", encoding="utf-8")
    completed = subprocess.run(
        ["bash", "-c", 'set -euo pipefail\n' + prologue + '\n: > "$TMPDIR/tmpxft_regression"\n'],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "TMPDIR": str(blocked / "job-tmp")},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (fallback / "tmpxft_regression").is_file()
    assert completed.stderr.count("[container-tmpdir]") == 1


def _run_guard(tmp_path, fallback, inherited=None):
    env = dict(os.environ)
    env.pop("TMPDIR", None)
    if inherited is not None:
        env["TMPDIR"] = str(inherited)
    return subprocess.run(
        [
            "bash", "-c",
            'set -euo pipefail\n' + common.container_tmpdir_guard(str(fallback))
            + '\n: > "$TMPDIR/tmpxft_regression"\nprintf "%s\\n" "$TMPDIR"\n',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


@pytest.mark.parametrize("inherited", [None, ""])
def test_unset_or_empty_tmpdir_uses_fallback(tmp_path, inherited):
    fallback = tmp_path / "fallback"
    completed = _run_guard(tmp_path, fallback, inherited)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(fallback)
    assert (fallback / "tmpxft_regression").is_file()
    assert completed.stderr.count("[container-tmpdir]") == 1


@pytest.mark.parametrize("exists", [True, False])
def test_usable_inherited_tmpdir_is_preserved_without_logging(tmp_path, exists):
    inherited = tmp_path / "job scratch's $HOME"
    if exists:
        inherited.mkdir()
    fallback = tmp_path / "fallback"
    completed = _run_guard(tmp_path, fallback, inherited)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(inherited)
    assert (inherited / "tmpxft_regression").is_file()
    assert completed.stderr == ""
    assert not fallback.exists()


@pytest.mark.parametrize("mode", [0o555, 0o666])
def test_read_only_or_unsearchable_tmpdir_uses_fallback(tmp_path, mode):
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    inherited = tmp_path / "inaccessible"
    inherited.mkdir(mode=mode)
    fallback = tmp_path / "fallback"
    try:
        completed = _run_guard(tmp_path, fallback, inherited)
    finally:
        inherited.chmod(0o755)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(fallback)
    assert (fallback / "tmpxft_regression").is_file()
    assert completed.stderr.count("[container-tmpdir]") == 1


@pytest.mark.parametrize("kind", ["file", "read-only", "unsearchable"])
def test_unusable_fallback_fails_before_native_work(tmp_path, kind):
    fallback = tmp_path / "unusable"
    if kind == "file":
        fallback.write_text("not a directory", encoding="utf-8")
    else:
        if os.geteuid() == 0:
            pytest.skip("root bypasses directory permissions")
        fallback.mkdir(mode=0o555 if kind == "read-only" else 0o666)
    try:
        completed = _run_guard(tmp_path, fallback)
    finally:
        fallback.chmod(0o755)
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "[container-tmpdir] fallback" in completed.stderr
    assert "not writable in the container" in completed.stderr
    assert not (fallback / "tmpxft_regression").exists()


def test_fallback_path_is_literal_shell_data(tmp_path):
    fallback = tmp_path / 'scratch space\'s "$HOME" `touch INJECTED` $(touch INJECTED) {cache}'
    completed = _run_guard(tmp_path, fallback)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(fallback)
    assert (fallback / "tmpxft_regression").is_file()
    assert f"using {fallback}" in completed.stderr
    assert not (tmp_path / "INJECTED").exists()


def test_every_container_rule_starts_with_the_rendered_guard(tmp_path):
    """Both feature modes together render every container rule, including reports."""
    source = (_REPOSITORY / "workflow" / "Snakefile").read_text(encoding="utf-8")
    rules = re.split(r"^[ \t]*rule (\w+):", source, flags=re.MULTILINE)
    container_rules = {
        name for name, body in zip(rules[1::2], rules[2::2])
        if re.search(r"^[ \t]*container:", body, re.MULTILINE)
    }
    rendered = set()
    for local_mmseqs in (False, True):
        case = tmp_path / str(local_mmseqs)
        data = case / "output" / "data"
        data.mkdir(parents=True)
        for protein in ("A", "B"):
            (data / f"{protein}.fasta").write_text(f">{protein}\nACDE\n", encoding="utf-8")
        fallback = case / "scratch's $HOME {cache}"
        extra = {
            "feature_directory": [],
            "enable_structure_analysis": True,
            "generate_recursive_report": True,
            "container_tmpdir": str(fallback),
            "mmseqs2_features": {"enabled": local_mmseqs},
        }
        if local_mmseqs:
            database_dir = case / "databases"
            database_dir.mkdir()
            databases = {}
            for name in ("uniref90", "mgnify", "small_bfd", "uniprot"):
                (database_dir / f"{name}.index").write_text("0\t0\t4\n", encoding="utf-8")
                databases[name] = {"path": str(database_dir / name), "identifier": f"test-{name}"}
            extra["mmseqs2_features"].update(
                temp_dir=str(case / "mmseqs-tmp"),
                template_database_ids={"pdb_seqres": "test-pdb-seqres", "mmcif": "test-mmcif"},
                databases=databases,
            )
        for job in _dry_run(case, **extra):
            name = re.match(r"(?:local)?rule (\w+):", job).group(1)
            if name not in container_rules or "Shell command:" not in job:
                continue
            shell = job.split("Shell command:", 1)[1].lstrip()
            # The local-MMseqs create_features variant is a Python symlink rule.
            if shell.splitlines()[0] == "None":
                continue
            assert shell.startswith(common.container_tmpdir_guard(str(fallback))), name
            rendered.add(name)
    assert rendered == container_rules
