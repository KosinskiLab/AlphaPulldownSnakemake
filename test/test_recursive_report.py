"""The recursive report reuses each fold's AlphaJudge table and is sized for its workers.

The helpers come from ``workflow/rules/common.smk``, loaded directly. The dry run
renders the real rule with the resident-batch fixtures; no container, GPU,
database, or cluster allocation is needed.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
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

ANALYSIS = {"--contact_thresh": 8.0, "--pae_filter": 100.0, "--models_to_analyse": "all"}


def test_report_scores_with_the_analysis_settings():
    assert common.recursive_report_arguments(ANALYSIS, {}, 4) == {**ANALYSIS, "--cores": 4}


def test_report_flags_extend_or_override_the_analysis_settings():
    report = {"--aggregate_report": "/reports/all.pdf", "--pae_filter": 50.0, "--cores": 2}
    assert common.recursive_report_arguments(ANALYSIS, report, 4) == {
        "--contact_thresh": 8.0,
        "--pae_filter": 50.0,
        "--models_to_analyse": "all",
        "--aggregate_report": "/reports/all.pdf",
        "--cores": 2,
    }


@pytest.mark.parametrize("report", [["--models_to_analyse=best"], "--models_to_analyse=best"])
def test_list_or_string_report_arguments_are_used_as_written(report):
    assert common.recursive_report_arguments(ANALYSIS, report, 4) == report


@pytest.mark.parametrize("tokens,peak_mb", [(6836, 5866), (6910, 5912), (7164, 6264)])
def test_run_memory_exceeds_measured_alphajudge_peaks(tokens, peak_mb):
    # The AF3 runs closest to the bound in a 1,900-run campaign.
    assert common.alphajudge_run_mem_mb(tokens) >= 1.25 * peak_mb


def test_report_memory_holds_the_largest_run_and_median_runs():
    tokens = [9000, 1000, 3000, 2000]
    largest, median = common.alphajudge_run_mem_mb(9000), common.alphajudge_run_mem_mb(3000)
    assert common.recursive_report_mem_mb(tokens, workers=4, attempt=1) == int(largest + 3 * median)
    assert common.recursive_report_mem_mb(tokens, workers=4, attempt=2) == int(1.5 * (largest + 3 * median))
    assert common.recursive_report_mem_mb(tokens, workers=4, attempt=1, cap_mb=10000) == 10000
    assert common.recursive_report_mem_mb(tokens, workers=1, attempt=1) == int(largest)


def test_report_memory_without_lengths_is_the_base_per_worker():
    assert common.recursive_report_mem_mb([], workers=4, attempt=1) == 4 * common.ALPHAJUDGE_BASE_MB
    assert common.recursive_report_mem_mb([0, 0], workers=4, attempt=1) == 4 * common.ALPHAJUDGE_BASE_MB


def test_report_command_uses_the_analysis_settings_and_its_threads(tmp_path):
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        pytest.skip("Snakemake executable is not available")
    config = yaml.safe_load((_FIXTURES / "af3.yaml").read_text(encoding="utf-8"))
    config.update(
        output_directory=str(tmp_path / "output"),
        structure_inference_unified_memory=False,
        enable_structure_analysis=True,
        generate_recursive_report=True,
        analyze_structure_arguments=ANALYSIS,
        recursive_report_arguments={"--aggregate_report": "/reports/all.pdf"},
        recursive_report_threads=3,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    completed = subprocess.run(
        [
            snakemake,
            "--snakefile", str(_REPOSITORY / "workflow" / "Snakefile"),
            "--configfile", str(config_path),
            "--cores", "4",
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
    job, = [
        job for job in re.split(r"^(?=(?:local)?rule \w+:$)", completed.stdout, flags=re.MULTILINE)
        if job.startswith("rule recursive_report:")
    ]
    assert re.search(r"^\s*threads: 3$", job, re.MULTILINE), job
    shell = job.split("Shell command:", 1)[1]
    for flag in (
        "--contact_thresh=8.0",
        "--pae_filter=100.0",
        "--models_to_analyse=all",
        "--aggregate_report=/reports/all.pdf",
        "--cores=3",
    ):
        assert flag in shell, (flag, shell)


@pytest.mark.parametrize("source", ["fasta", "af3_json", "direct_af3_json"])
def test_report_memory_uses_staged_lengths_without_length_filter(tmp_path, source):
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        pytest.skip("Snakemake executable is not available")
    output = tmp_path / "output"
    features = output / "features"
    data = output / "data"
    features.mkdir(parents=True)
    data.mkdir()
    tokens = 9000
    name = "Large.json" if source == "direct_af3_json" else "Large"
    sheet = tmp_path / "folds.csv"
    sheet.write_text(name + "\n", encoding="utf-8")
    if source == "fasta":
        (data / "Large.fasta").write_text(">Large\n" + "A" * tokens + "\n", encoding="utf-8")
    feature_name = "Large.json" if source == "direct_af3_json" else "Large_af3_input.json"
    (features / feature_name).write_text(json.dumps({
        "sequences": [{"protein": {"id": "A", "sequence": "A" * tokens}}]
    }), encoding="utf-8")
    config = yaml.safe_load((_FIXTURES / "af3.yaml").read_text(encoding="utf-8"))
    config.update(
        input_files=[str(sheet)], output_directory=str(output),
        feature_directory=[str(features)], batch_size=1,
        max_total_length=0, max_protein_length=0, length_filter_fetch_uniprot=False,
        structure_inference_unified_memory=False,
        enable_structure_analysis=True, generate_recursive_report=True,
        analyze_structure_arguments=ANALYSIS, recursive_report_threads=2,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    completed = subprocess.run(
        [snakemake, "--snakefile", str(_REPOSITORY / "workflow" / "Snakefile"),
         "--configfile", str(config_path), "--cores", "2", "--rerun-triggers", "mtime",
         "--touch", "--printshellcmds"],
        cwd=_REPOSITORY, capture_output=True, text=True, timeout=120,
        env={**os.environ, "APPTAINER_BINDPATH": "", "SINGULARITY_BINDPATH": ""},
    )
    log = completed.stdout + completed.stderr
    assert completed.returncode == 0, log[-4000:]
    job, = [job for job in re.split(r"^(?=(?:local)?rule \w+:$)", log, flags=re.MULTILINE)
            if job.startswith(("rule recursive_report:", "localrule recursive_report:"))]
    expected = common.recursive_report_mem_mb([tokens], workers=2, attempt=1)
    assert int(re.search(r"\bmem_mb=(\d+)", job).group(1)) == expected
