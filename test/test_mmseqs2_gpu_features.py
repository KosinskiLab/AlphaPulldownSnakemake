"""Behavioral tests for the opt-in local MMseqs2-GPU workflow adapter."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


_MODULE = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "rules"
    / "mmseqs2_gpu.smk"
)
_loader = importlib.machinery.SourceFileLoader("aps_mmseqs2_gpu", str(_MODULE))
_spec = importlib.util.spec_from_loader("aps_mmseqs2_gpu", _loader)
mmseqs2_gpu = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mmseqs2_gpu
_loader.exec_module(mmseqs2_gpu)


def _config(**overrides):
    config = {
        "enabled": True,
        "temp_dir": "/scratch/mmseqs",
        "batch_max_sequences": 2,
        "batch_max_residues": 9,
        "databases": {
            name: {
                "path": f"/db/{name}",
                "identifier": f"{name}-2026-08",
            }
            for name in ("uniref90", "mgnify", "small_bfd", "uniprot")
        },
    }
    config.update(overrides)
    return config


def test_enabled_af3_adapter_preserves_deep_interface_chunk_limits():
    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(), data_pipeline="alphafold3"
    )

    assert adapter.batch_max_sequences == 2
    assert adapter.batch_max_residues == 9


def test_local_mmseqs_features_are_af3_only_and_require_explicit_databases():
    with pytest.raises(ValueError, match="AlphaFold 3"):
        mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
            _config(), data_pipeline="alphafold2"
        )

    missing = _config()
    del missing["databases"]["uniprot"]
    with pytest.raises(ValueError, match="uniprot"):
        mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
            missing, data_pipeline="alphafold3"
        )


def test_cli_arguments_pass_every_explicit_search_input_to_alphapulldown():
    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(), data_pipeline="af3"
    )

    arguments = adapter.cli_arguments(threads=12)

    assert "--mmseqs_binary_path=/opt/mmseqs/bin/mmseqs" in arguments
    assert "--mmseqs_temp_dir=/scratch/mmseqs" in arguments
    assert "--mmseqs_batch_max_sequences=2" in arguments
    assert "--mmseqs_batch_max_residues=9" in arguments
    assert "--mmseqs_threads=12" in arguments
    for name in ("uniref90", "mgnify", "small_bfd", "uniprot"):
        assert f"--mmseqs_{name}_database_path=/db/{name}" in arguments
        assert f"--mmseqs_{name}_database_id={name}-2026-08" in arguments

    assert adapter.bind_paths == (Path("/scratch/mmseqs"), Path("/db"))


def test_bind_paths_include_resolved_database_targets_but_not_binary(tmp_path):
    binary_links = tmp_path / "binary-links"
    binary_links.mkdir()
    database_links = tmp_path / "database-links"
    database_links.mkdir()
    binary_target = tmp_path / "installation" / "bin" / "mmseqs"
    binary_target.parent.mkdir(parents=True)
    binary_target.touch()
    binary_link = binary_links / "mmseqs"
    binary_link.symlink_to(binary_target)
    database_target = tmp_path / "storage" / "uniref90"
    database_target.parent.mkdir()
    database_target.touch()
    database_link = database_links / "uniref90"
    database_link.symlink_to(database_target)
    config = _config(binary_path=str(binary_link))
    config["temp_dir"] = str(tmp_path / "scratch")
    config["databases"]["uniref90"]["path"] = str(database_link)

    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        config, data_pipeline="alphafold3"
    )

    assert binary_target.parent not in adapter.bind_paths
    assert binary_link.parent not in adapter.bind_paths
    assert database_target.parent in adapter.bind_paths


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("batch_max_sequences", 0),
        ("batch_max_residues", 0),
        ("sensitivity", 0),
        ("e_value", 0),
    ),
)
def test_invalid_search_limits_fail_during_workflow_parsing(field, value):
    with pytest.raises(ValueError, match=field):
        mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
            _config(**{field: value}), data_pipeline="alphafold3"
        )


def test_invalid_database_hit_limit_fails_during_workflow_parsing():
    config = _config()
    config["databases"]["uniprot"]["max_sequences"] = 0

    with pytest.raises(ValueError, match="uniprot.*max_sequences"):
        mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
            config, data_pipeline="alphafold3"
        )


def test_missing_cache_artifact_schedules_batch_despite_existing_sentinel(tmp_path):
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        pytest.skip("Snakemake executable is not available")

    repository = Path(__file__).resolve().parents[1]
    sample_sheet = tmp_path / "sample_sheet.csv"
    sample_sheet.write_text("alpha\n", encoding="utf-8")
    output_directory = tmp_path / "output"
    data_directory = output_directory / "data"
    feature_directory = output_directory / "features"
    data_directory.mkdir(parents=True)
    feature_directory.mkdir()
    (data_directory / "alpha.fasta").write_text(
        ">alpha\nACDEFG\n", encoding="utf-8"
    )
    (feature_directory / ".mmseqs2_gpu.complete").touch()

    with (repository / "config" / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        {
            "input_files": [str(sample_sheet)],
            "output_directory": str(output_directory),
            "feature_directory": [],
            "only_generate_features": True,
            "enable_structure_analysis": False,
            "max_total_length": 0,
            "mmseqs2_gpu_features": _config(),
        }
    )
    config["create_feature_arguments"]["--compress_features"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [
            snakemake,
            "--snakefile",
            str(repository / "workflow" / "Snakefile"),
            "--configfile",
            str(config_path),
            "--dry-run",
            "--cores",
            "1",
            "--rerun-triggers",
            "mtime",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "APPTAINER_BINDPATH": "", "SINGULARITY_BINDPATH": ""},
    )

    assert "create_features_mmseqs2_gpu" in completed.stdout
    cache_artifact = feature_directory / ".mmseqs2_gpu_cache" / "alpha_af3_input.json"
    assert str(cache_artifact) in completed.stdout
