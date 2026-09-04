"""Behavioral tests for the opt-in local MMseqs2-GPU workflow adapter."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import hashlib
import json
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
_REPOSITORY = Path(__file__).resolve().parents[1]


def _config(**overrides):
    config = {
        "enabled": True,
        "temp_dir": "/scratch/mmseqs",
        "batch_max_sequences": 2,
        "batch_max_residues": 9,
        "template_database_ids": {
            "pdb_seqres": "pdb-seqres-2026-08",
            "mmcif": "pdb-mmcif-2026-08",
        },
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


def _workflow_case(tmp_path, proteins, *, mmseqs_config=None):
    sample_sheet = tmp_path / "sample_sheet.csv"
    sample_sheet.write_text("+".join(proteins) + "\n", encoding="utf-8")
    output_directory = tmp_path / "output"
    data_directory = output_directory / "data"
    data_directory.mkdir(parents=True)
    (output_directory / "features").mkdir()
    for protein in proteins:
        (data_directory / f"{protein}.fasta").write_text(
            f">{protein}\nACDE\n", encoding="utf-8"
        )
    with (_REPOSITORY / "config" / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        {
            "input_files": [str(sample_sheet)],
            "output_directory": str(output_directory),
            "feature_directory": [],
            "only_generate_features": True,
            "enable_structure_analysis": False,
            "max_total_length": 0,
            "mmseqs2_features": mmseqs_config or _config(),
        }
    )
    config["create_feature_arguments"]["--compress_features"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, config_path, output_directory


def _run_snakemake(config_path, *targets, check=True, extra_args=(), env=None):
    snakemake = shutil.which("snakemake")
    if snakemake is None:
        pytest.skip("Snakemake executable is not available")
    return subprocess.run(
        [
            snakemake,
            "--snakefile",
            str(_REPOSITORY / "workflow" / "Snakefile"),
            *map(str, targets),
            "--configfile",
            str(config_path),
            "--cores",
            "1",
            "--rerun-triggers",
            "mtime",
            *extra_args,
        ],
        cwd=_REPOSITORY,
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
        env=env
        or {**os.environ, "APPTAINER_BINDPATH": "", "SINGULARITY_BINDPATH": ""},
    )


def _completion_summary(protein, bundle):
    stat = bundle.stat()
    return {
        "schemaVersion": 2,
        "artifacts": [
            {
                "name": protein,
                "file": bundle.name,
                "sizeBytes": stat.st_size,
                "mtimeNs": stat.st_mtime_ns,
                "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            }
        ],
        "reused": [],
        "written": [protein],
    }


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

    arguments = adapter.msa_cli_arguments(threads=12, memory_mb=160000)

    assert "--mmseqs_binary_path=/opt/mmseqs/bin/mmseqs" in arguments
    assert "--mmseqs_temp_dir=/scratch/mmseqs" in arguments
    assert "--mmseqs_batch_max_sequences=2" in arguments
    assert "--mmseqs_batch_max_residues=9" in arguments
    assert "--mmseqs_threads=12" in arguments
    assert not any("sensitivity" in argument for argument in arguments)
    for name in ("uniref90", "mgnify", "small_bfd", "uniprot"):
        assert f"--mmseqs_{name}_database_path=/db/{name}" in arguments
        assert f"--mmseqs_{name}_database_id={name}-2026-08" in arguments

    assert adapter.bind_paths == (Path("/scratch/mmseqs"), Path("/db"))
    assert adapter.binary_id == "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"


def test_finalize_arguments_include_explicit_template_provenance():
    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(), data_pipeline="af3"
    )

    arguments = adapter.finalize_cli_arguments()

    assert "--template_seqres_database_id=pdb-seqres-2026-08" in arguments
    assert "--template_mmcif_database_id=pdb-mmcif-2026-08" in arguments


def test_non_bundled_binary_is_rejected_instead_of_becoming_invisible(tmp_path):
    binary_links = tmp_path / "binary-links"
    binary_links.mkdir()
    binary_target = tmp_path / "installation" / "bin" / "mmseqs"
    binary_target.parent.mkdir(parents=True)
    binary_target.touch()
    binary_link = binary_links / "mmseqs"
    binary_link.symlink_to(binary_target)
    config = _config(binary_path=str(binary_link))

    with pytest.raises(ValueError, match="bundled.*binary"):
        mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
            config, data_pipeline="alphafold3"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("batch_max_sequences", 0),
        ("batch_max_residues", 0),
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


def test_missing_template_database_identity_fails_during_workflow_parsing():
    config = _config()
    del config["template_database_ids"]["mmcif"]

    with pytest.raises(ValueError, match="mmcif"):
        mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
            config, data_pipeline="alphafold3"
        )


def test_feature_requests_are_split_into_stable_bounded_gpu_shards():
    shards = mmseqs2_gpu.plan_feature_shards(
        ("a", "b", "c", "d", "e"),
        {"a": 4, "b": 4, "c": 8, "d": 1, "e": 1},
        max_sequences=2,
        max_residues=9,
    )

    assert [shard.proteins for shard in shards] == [
        ("a", "b"),
        ("c", "d"),
        ("e",),
    ]
    assert [shard.total_residues for shard in shards] == [8, 9, 1]
    assert len({shard.identifier for shard in shards}) == 3


def test_unknown_length_requests_run_alone_so_one_shard_is_one_core_chunk():
    shards = mmseqs2_gpu.plan_feature_shards(
        ("known-a", "unknown", "known-b"),
        {"known-a": 4, "unknown": 0, "known-b": 4},
        max_sequences=10,
        max_residues=9,
    )

    assert [shard.proteins for shard in shards] == [
        ("known-a",),
        ("unknown",),
        ("known-b",),
    ]


def test_repair_job_identity_changes_with_missing_bundle_state():
    shard = mmseqs2_gpu.FeatureShard("0000-base", ("alpha", "beta"), 8)

    first = mmseqs2_gpu.repair_shard_identifier(
        shard, ("beta",), cache_mtime_ns=100
    )
    repeated_loss = mmseqs2_gpu.repair_shard_identifier(
        shard, ("beta",), cache_mtime_ns=200
    )

    assert first.startswith("0000-base-repair-")
    assert first != repeated_loss


def test_unchanged_bundle_uses_manifest_stat_fast_path(tmp_path, monkeypatch):
    shard = mmseqs2_gpu.FeatureShard("0000-base", ("alpha",), 4)
    bundle = tmp_path / "alpha_mmseqs_msa.json"
    bundle.write_bytes(b'{"sequence":"ACDE"}\n')
    summary_dir = tmp_path / ".completed"
    summary_dir.mkdir()
    summary_path = summary_dir / "0000-base.json"
    summary_path.write_text(
        json.dumps(_completion_summary("alpha", bundle)), encoding="utf-8"
    )
    monkeypatch.setattr(
        mmseqs2_gpu,
        "_file_sha256",
        lambda path: pytest.fail("unchanged bundles must not be rehashed"),
    )

    scheduled = mmseqs2_gpu.schedule_feature_shards((shard,), tmp_path)[0]

    assert scheduled.identifier == shard.identifier


def test_changed_mtime_with_matching_digest_remains_valid(tmp_path):
    shard = mmseqs2_gpu.FeatureShard("0000-base", ("alpha",), 4)
    bundle = tmp_path / "alpha_mmseqs_msa.json"
    bundle.write_bytes(b'{"sequence":"ACDE"}\n')
    summary_dir = tmp_path / ".completed"
    summary_dir.mkdir()
    summary_path = summary_dir / "0000-base.json"
    summary_path.write_text(
        json.dumps(_completion_summary("alpha", bundle)), encoding="utf-8"
    )
    current = bundle.stat()
    os.utime(
        bundle,
        ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
    )

    scheduled = mmseqs2_gpu.schedule_feature_shards((shard,), tmp_path)[0]

    assert scheduled.identifier == shard.identifier


def test_completed_shard_is_repaired_when_bundle_digest_no_longer_matches(tmp_path):
    shard = mmseqs2_gpu.FeatureShard("0000-base", ("alpha",), 4)
    bundle = tmp_path / "alpha_mmseqs_msa.json"
    original = b'{"sequence":"ACDE"}\n'
    bundle.write_bytes(original)
    summary_dir = tmp_path / ".completed"
    summary_dir.mkdir()
    (summary_dir / "0000-base.json").write_text(
        json.dumps(_completion_summary("alpha", bundle)),
        encoding="utf-8",
    )

    valid = mmseqs2_gpu.schedule_feature_shards((shard,), tmp_path)[0]
    previous = bundle.stat()
    bundle.write_text('{"sequence":"WXYZ"}\n', encoding="utf-8")
    os.utime(
        bundle,
        ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000),
    )
    corrupt = mmseqs2_gpu.schedule_feature_shards((shard,), tmp_path)[0]

    assert valid.identifier == shard.identifier
    assert corrupt.identifier.startswith(f"{shard.identifier}-repair-")

    repaired_bytes = b'{"sequence":"ACDE","repaired":true}\n'
    bundle.write_bytes(repaired_bytes)
    corrupt.summary_path.write_text(
        json.dumps(_completion_summary("alpha", bundle)),
        encoding="utf-8",
    )

    repaired = mmseqs2_gpu.schedule_feature_shards((shard,), tmp_path)[0]
    assert repaired.identifier == corrupt.identifier


@pytest.mark.parametrize("summary", ([], "complete", 1, None))
def test_non_object_completion_summary_schedules_repair(tmp_path, summary):
    shard = mmseqs2_gpu.FeatureShard("0000-base", ("alpha",), 4)
    summary_dir = tmp_path / ".completed"
    summary_dir.mkdir()
    (summary_dir / "0000-base.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    scheduled = mmseqs2_gpu.schedule_feature_shards((shard,), tmp_path)[0]

    assert scheduled.identifier.startswith(f"{shard.identifier}-repair-")


def test_search_memory_is_flat_in_shard_size():
    """Measured: 1, 8, 32 and 128 queries against the full database set all peaked at
    the same 149 GB, so the estimate must not grow with the query load."""
    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(batch_max_residues=100_000, search_ram_mb=160_000, gpu_ram_scaling=1.2),
        data_pipeline="alphafold3",
    )
    assert adapter.search_memory_mb(safety=1.0, attempt=1) == 160_000
    assert adapter.search_memory_mb(safety=1.0, attempt=2) == 192_000


def test_cpu_search_is_allowed_more_wall_time_than_gpu():
    """CPU search measured 2.4x slower than GPU on the same shard."""
    gpu = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(search_runtime_base_minutes=90), data_pipeline="alphafold3"
    )
    cpu = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(search_runtime_base_minutes=90, use_gpu=False),
        data_pipeline="alphafold3",
    )
    assert gpu.use_gpu is True and cpu.use_gpu is False
    assert cpu.search_runtime_minutes(8, 1_000, attempt=1) > (
        gpu.search_runtime_minutes(8, 1_000, attempt=1)
    )


def test_finalization_is_sized_far_below_msa_generation():
    """Measured 0.24 GB / 13 s for one protein; the old model asked for 24 GB and 24 h."""
    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(), data_pipeline="alphafold3"
    )
    assert adapter.finalize_memory_mb(400, safety=1.0, attempt=1) < 6_000
    assert adapter.finalize_runtime_minutes(attempt=1) <= 60


def test_use_gpu_reaches_the_core_command():
    for use_gpu, expected in ((True, "true"), (False, "false")):
        adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
            _config(use_gpu=use_gpu), data_pipeline="alphafold3"
        )
        args = adapter.msa_cli_arguments(threads=8, memory_mb=160000)
        assert f"--mmseqs_use_gpu={expected}" in args


def test_cache_namespaces_change_with_search_and_template_provenance():
    first = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(), data_pipeline="alphafold3"
    )
    changed_search = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(e_value=1e-5), data_pipeline="alphafold3"
    )
    changed_templates_config = _config()
    changed_templates_config["template_database_ids"]["mmcif"] = "pdb-mmcif-new"
    changed_templates = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        changed_templates_config, data_pipeline="alphafold3"
    )

    assert first.msa_cache_key("image:v1") != changed_search.msa_cache_key("image:v1")
    assert first.feature_cache_key("2050-01-01", "image:v1") != (
        changed_templates.feature_cache_key("2050-01-01", "image:v1")
    )

    moved_config = _config()
    moved_config["databases"]["uniref90"]["path"] = "/new/mount/uniref90"
    moved = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        moved_config, data_pipeline="alphafold3"
    )
    assert first.msa_cache_key("image:v1") == moved.msa_cache_key("image:v1")

    changed_binary = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(binary_id="new-mmseqs-build"), data_pipeline="alphafold3"
    )
    assert first.msa_cache_key("image:v1") != changed_binary.msa_cache_key("image:v1")


def test_partial_msa_cache_schedules_shard_retry_and_cpu_finalization(tmp_path):
    config, config_path, output_directory = _workflow_case(tmp_path, ("alpha",))
    feature_directory = output_directory / "features"
    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        config["mmseqs2_features"], data_pipeline="alphafold3"
    )
    msa_cache = (
        feature_directory
        / ".mmseqs2_gpu_msa_cache"
        / adapter.msa_cache_key(config["prediction_container"])
    )
    msa_cache.mkdir(parents=True)
    partial_bundle = msa_cache / "alpha_mmseqs_msa.json"
    partial_bundle.write_text('{"durable": true}\n', encoding="utf-8")
    completed = _run_snakemake(config_path, extra_args=("--dry-run",))

    assert "create_mmseqs2_gpu_msa_shard" in completed.stdout
    assert "finalize_mmseqs2_features" in completed.stdout
    assert partial_bundle.read_text(encoding="utf-8") == '{"durable": true}\n'
    cache_artifact = (
        feature_directory
        / ".mmseqs2_gpu_cache"
        / adapter.feature_cache_key(
            config["create_feature_arguments"]["--max_template_date"],
            config["prediction_container"],
        )
        / "alpha_af3_input.json"
    )
    assert str(cache_artifact) in completed.stdout


def test_missing_bundle_after_completion_schedules_automatic_gpu_repair(tmp_path):
    config, config_path, output_directory = _workflow_case(tmp_path, ("alpha",))
    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        config["mmseqs2_features"], data_pipeline="alphafold3"
    )
    msa_cache = (
        output_directory
        / "features"
        / ".mmseqs2_gpu_msa_cache"
        / adapter.msa_cache_key(config["prediction_container"])
    )
    shard = mmseqs2_gpu.plan_feature_shards(
        ("alpha",),
        {"alpha": 4},
        max_sequences=2,
        max_residues=9,
    )[0]
    base_summary = msa_cache / ".completed" / f"{shard.identifier}.json"
    base_summary.parent.mkdir(parents=True)
    base_summary.write_text('{"status": "complete"}\n', encoding="utf-8")
    completed = _run_snakemake(config_path, extra_args=("--dry-run",))

    assert "create_mmseqs2_gpu_msa_shard" in completed.stdout
    assert f"{shard.identifier}-repair-" in completed.stdout


def test_dry_run_schedules_bounded_gpu_shards_then_parallel_cpu_jobs(tmp_path):
    proteins = ["alpha", "beta", "gamma"]
    _, config_path, _ = _workflow_case(
        tmp_path,
        proteins,
        mmseqs_config=_config(
            batch_max_sequences=2,
            batch_max_residues=1_000,
        ),
    )
    completed = _run_snakemake(config_path, extra_args=("--dry-run",))

    assert completed.stdout.count("rule create_mmseqs2_gpu_msa_shard:") == 2
    assert completed.stdout.count("rule finalize_mmseqs2_features:") == 3
    assert "gpu=1" in completed.stdout


def test_failed_gpu_shard_preserves_completed_msa_bundles_for_retry(tmp_path):
    proteins = ("alpha", "beta")
    config, config_path, output_directory = _workflow_case(
        tmp_path,
        proteins,
        mmseqs_config=_config(
            batch_max_sequences=2,
            batch_max_residues=1_000,
        ),
    )

    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        config["mmseqs2_features"], data_pipeline="alphafold3"
    )
    msa_cache = (
        output_directory
        / "features"
        / ".mmseqs2_gpu_msa_cache"
        / adapter.msa_cache_key(config["prediction_container"])
    )
    shard = mmseqs2_gpu.plan_feature_shards(
        proteins,
        {protein: 4 for protein in proteins},
        max_sequences=2,
        max_residues=1_000,
    )[0]
    summary = msa_cache / ".completed" / f"{shard.identifier}.json"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_cli = fake_bin / "create_batch_msas.py"
    fake_cli.write_text(
        """#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fasta_paths")
parser.add_argument("--msa_output_dir")
parser.add_argument("--summary_path")
args, _ = parser.parse_known_args()
output = Path(args.msa_output_dir)
output.mkdir(parents=True, exist_ok=True)
fastas = [Path(path) for path in args.fasta_paths.split(",")]
attempt = output / ".fake_attempt"
first = output / (fastas[0].stem + "_mmseqs_msa.json")
if not attempt.exists():
    first.write_text(json.dumps({"protein": fastas[0].stem}) + "\\n")
    attempt.write_text("failed once\\n")
    raise SystemExit(1)
if first.exists():
    (output / ".reused").write_text(first.name + "\\n")
for fasta in fastas:
    bundle = output / (fasta.stem + "_mmseqs_msa.json")
    if not bundle.exists():
        bundle.write_text(json.dumps({"protein": fasta.stem}) + "\\n")
summary = Path(args.summary_path)
summary.parent.mkdir(parents=True, exist_ok=True)
artifacts = []
for fasta in fastas:
    bundle = output / (fasta.stem + "_mmseqs_msa.json")
    stat = bundle.stat()
    import hashlib
    artifacts.append({
        "name": fasta.stem,
        "file": bundle.name,
        "sizeBytes": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
        "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
    })
summary.write_text(json.dumps({
    "schemaVersion": 2,
    "artifacts": artifacts,
    "written": [fasta.stem for fasta in fastas],
    "reused": [],
}) + "\\n")
""",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "APPTAINER_BINDPATH": "",
        "SINGULARITY_BINDPATH": "",
    }

    failed = _run_snakemake(
        config_path,
        summary,
        check=False,
        extra_args=("--retries", "0"),
        env=environment,
    )

    assert failed.returncode != 0
    assert (msa_cache / "alpha_mmseqs_msa.json").exists()
    assert not summary.exists()

    _run_snakemake(
        config_path,
        summary,
        extra_args=("--retries", "0"),
        env=environment,
    )

    assert summary.exists()
    assert (msa_cache / "beta_mmseqs_msa.json").exists()
    assert (msa_cache / ".reused").read_text(encoding="utf-8") == (
        "alpha_mmseqs_msa.json\n"
    )

    settled = _run_snakemake(
        config_path,
        summary,
        extra_args=("--dry-run",),
        env=environment,
    )
    assert "Nothing to be done" in settled.stdout


def test_split_memory_limit_follows_the_slurm_allocation():
    """MMseqs2 otherwise sizes splits from physical node memory and ignores the cgroup,
    so on a large node with a small allocation it is OOM-killed instead of splitting."""
    adapter = mmseqs2_gpu.LocalMmseqsFeatureConfig.from_mapping(
        _config(), data_pipeline="alphafold3"
    )
    args = adapter.msa_cli_arguments(threads=8, memory_mb=120_000)
    limit = next(a for a in args if a.startswith("--mmseqs_split_memory_limit="))
    assert limit.split("=", 1)[1].strip("'\"") == "108000M"
