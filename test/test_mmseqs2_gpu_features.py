"""Behavioral tests for the opt-in local MMseqs2-GPU workflow adapter."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest


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
        "binary_path": "/opt/mmseqs/bin/mmseqs",
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

    assert adapter.bind_paths == (
        Path("/opt/mmseqs/bin"),
        Path("/scratch/mmseqs"),
        Path("/db"),
    )


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
