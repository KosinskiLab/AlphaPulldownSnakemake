"""Thin Snakemake adapter for batched local MMseqs2-GPU feature generation.

The AlphaPulldown package owns MSA generation and AF3 finalization. This module
owns the workflow-facing configuration, stable shard plan, resource model,
cache namespaces, and required container bind paths.
"""

import glob
import os
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shlex
from typing import Any, Mapping, Sequence


_DATABASE_NAMES = ("uniref90", "mgnify", "small_bfd", "uniprot")
# The three nucleotide references AlphaFold 3 merges into one unpaired RNA MSA.
# Optional and all-or-nothing: configure all three to search RNA chains, or none and
# the stage stays protein-only exactly as before.
_RNA_DATABASE_NAMES = ("rfam", "rnacentral", "nt_rna")
# Measured floor for a shard that searches the RNA databases. The protein estimate
# below scales the padded database size on disk by 0.85, but that ratio does not carry
# over to a plain nucleotide database: nt_rna is 77 GB on disk and 150 GB and 200 GB
# both failed, while 275 GB and 350 GB both succeeded and both peaked at 249 GB.
# MMseqs2 expands to fill whatever it is given -- at 500 GB the same search peaked at
# 489 GB -- so peak RSS is not a requirement and only the failure boundary is.
_RNA_SEARCH_FLOOR_MB = 275 * 1024
_BUNDLED_MMSEQS_BINARY = Path("/opt/mmseqs/bin/mmseqs")
_BUNDLED_MMSEQS_ID = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
_BACKENDS = {"alphafold3": "alphafold3", "af3": "alphafold3",
             "alphafold2": "alphafold2", "af2": "alphafold2"}
# create_feature_arguments an AlphaFold 2 finalization reads: the template stack,
# and nothing else. The MSA arguments (jackhmmer, HHblits, their databases) belong
# to the native search this stage replaces, so they are deliberately not forwarded.
_AF2_TEMPLATE_ARGUMENTS = (
    "--use_hhsearch",
    "--pdb_seqres_database_path",
    "--pdb70_database_path",
    "--template_mmcif_dir",
    "--obsolete_pdbs_path",
    "--hmmsearch_binary_path",
    "--hmmbuild_binary_path",
    "--hhsearch_binary_path",
    "--kalign_binary_path",
)


def _enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Local MMseqs2-GPU {context} requires {key!r}")
    return value


@dataclass(frozen=True)
class MmseqsDatabaseConfig:
    path: Path
    identifier: str
    max_sequences: int | None = None


@dataclass(frozen=True)
class FeatureShard:
    """A stable GPU scheduling unit containing one bounded MMseqs2 chunk."""

    identifier: str
    proteins: tuple[str, ...]
    total_residues: int


@dataclass(frozen=True)
class ScheduledShard:
    """A shard paired with the completion target selected for this DAG."""

    identifier: str
    shard: FeatureShard
    summary_path: Path


def plan_feature_shards(
    proteins: Sequence[str],
    sequence_lengths: Mapping[str, int],
    *,
    max_sequences: int,
    max_residues: int,
) -> tuple[FeatureShard, ...]:
    """Split requests into deterministic shards bounded by count and residues.

    Unknown-length requests run alone. The core batch interface still enforces
    both limits after reading the FASTAs, so an input downloaded after DAG
    construction remains safe and still maps to one core chunk.
    """
    groups: list[tuple[tuple[str, ...], int]] = []
    current: list[str] = []
    current_residues = 0
    for protein in proteins:
        residues = max(int(sequence_lengths.get(protein, 0) or 0), 0)
        if residues == 0:
            if current:
                groups.append((tuple(current), current_residues))
                current = []
                current_residues = 0
            groups.append(((protein,), 0))
            continue
        over_count = len(current) >= max_sequences
        over_residues = bool(
            current and residues and current_residues + residues > max_residues
        )
        if over_count or over_residues:
            groups.append((tuple(current), current_residues))
            current = []
            current_residues = 0
        current.append(protein)
        current_residues += residues
    if current:
        groups.append((tuple(current), current_residues))

    shards = []
    for index, (members, residues) in enumerate(groups):
        digest = hashlib.sha256("\0".join(members).encode()).hexdigest()[:12]
        shards.append(
            FeatureShard(
                identifier=f"{index:04d}-{digest}",
                proteins=members,
                total_residues=residues,
            )
        )
    return tuple(shards)


def repair_shard_identifier(
    shard: FeatureShard,
    invalid_state: Sequence[str],
    *,
    cache_mtime_ns: int,
) -> str:
    """Return a fresh job id when a completed shard has invalid cached bundles."""
    state = "\0".join((*invalid_state, str(cache_mtime_ns)))
    digest = hashlib.sha256(state.encode()).hexdigest()[:12]
    return f"{shard.identifier}-repair-{digest}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_shard_summary(
    summary_path: Path, shard: FeatureShard, msa_cache_dir: Path
) -> tuple[bool, tuple[str, ...]]:
    """Validate a completion manifest without normally rereading MSA bundles.

    The writer records the final file stat with its digest. An unchanged stat is
    the common fast path; metadata changes fall back to a streaming SHA-256 so a
    harmless move or touch does not force a GPU rerun.
    """
    try:
        payload = json.loads(summary_path.read_bytes())
    except (OSError, ValueError) as exc:
        return False, (f"summary:{type(exc).__name__}",)
    if not isinstance(payload, Mapping):
        return False, ("summary:not-object",)
    if payload.get("schemaVersion") != 2:
        return False, (f"summary-schema:{payload.get('schemaVersion')!r}",)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return False, ("summary-artifacts:missing",)
    records = {}
    for record in artifacts:
        if not isinstance(record, Mapping):
            return False, ("summary-artifacts:record",)
        name = record.get("name")
        if not isinstance(name, str) or name in records:
            return False, ("summary-artifacts:names",)
        records[name] = record
    expected_names = set(shard.proteins)
    if set(records) != expected_names:
        return False, ("summary-artifacts:names",)

    written = payload.get("written")
    reused = payload.get("reused")
    if not isinstance(written, list) or not isinstance(reused, list):
        return False, ("summary-results:missing",)
    if (
        not all(isinstance(name, str) for name in (*written, *reused))
        or set(written) & set(reused)
        or set(written) | set(reused) != expected_names
    ):
        return False, ("summary-results:names",)

    invalid = []
    for protein in shard.proteins:
        expected_file = f"{protein}_mmseqs_msa.json"
        record = records[protein]
        if record.get("file") != expected_file:
            invalid.append(f"{protein}:manifest")
            continue
        expected_digest = record.get("sha256")
        try:
            digest_is_valid = (
                isinstance(expected_digest, str)
                and len(expected_digest) == 64
                and int(expected_digest, 16) >= 0
            )
        except ValueError:
            digest_is_valid = False
        expected_size = record.get("sizeBytes")
        expected_mtime_ns = record.get("mtimeNs")
        if not digest_is_valid:
            invalid.append(f"{protein}:digest")
            continue
        if not isinstance(expected_size, int) or expected_size < 0:
            invalid.append(f"{protein}:size")
            continue
        if not isinstance(expected_mtime_ns, int) or expected_mtime_ns < 0:
            invalid.append(f"{protein}:mtime")
            continue
        bundle = msa_cache_dir / expected_file
        try:
            stat = bundle.stat()
        except OSError:
            invalid.append(f"{protein}:missing")
            continue
        if stat.st_size != expected_size:
            invalid.append(
                f"{protein}:size:{stat.st_size}:{stat.st_mtime_ns}"
            )
            continue
        if stat.st_mtime_ns == expected_mtime_ns:
            continue
        try:
            actual_digest = _file_sha256(bundle)
        except OSError:
            invalid.append(f"{protein}:missing")
            continue
        if actual_digest != expected_digest:
            invalid.append(
                f"{protein}:digest:{actual_digest}:{stat.st_mtime_ns}"
            )
    return not invalid, tuple(invalid)


def _mtime_ns(path: Path) -> int:
    """Return a sortable mtime while tolerating concurrent cache cleanup."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def schedule_feature_shards(
    shards: Sequence[FeatureShard], msa_cache_dir: Path
) -> tuple[ScheduledShard, ...]:
    """Select normal or repair completion targets from durable cache state.

    A completion summary with any invalid per-protein bundle is inconsistent.
    A fresh repair target makes Snakemake rerun that GPU shard while the core
    MSA batch reuses every surviving bundle.
    """
    scheduled = []
    completion_dir = msa_cache_dir / ".completed"
    for shard in shards:
        base_summary = completion_dir / f"{shard.identifier}.json"
        candidates = [base_summary]
        if completion_dir.exists():
            candidates.extend(
                completion_dir.glob(f"{shard.identifier}-repair-*.json")
            )
        candidates = sorted(
            {candidate for candidate in candidates if candidate.exists()},
            key=_mtime_ns,
            reverse=True,
        )
        valid_summary = None
        invalid_state = []
        for candidate in candidates:
            valid, state = validate_shard_summary(
                candidate, shard, msa_cache_dir
            )
            if valid:
                valid_summary = candidate
                break
            invalid_state.extend(state)

        if valid_summary is not None:
            job_id = valid_summary.stem
            summary_path = valid_summary
        elif not candidates:
            job_id = shard.identifier
            summary_path = base_summary
        else:
            try:
                cache_mtime_ns = msa_cache_dir.stat().st_mtime_ns
            except OSError:
                cache_mtime_ns = 0
            job_id = repair_shard_identifier(
                shard, invalid_state, cache_mtime_ns=cache_mtime_ns
            )
            summary_path = completion_dir / f"{job_id}.json"
        scheduled.append(
            ScheduledShard(
                identifier=job_id,
                shard=shard,
                summary_path=summary_path,
            )
        )
    return tuple(scheduled)


@dataclass(frozen=True)
class LocalMmseqsFeatureConfig:
    enabled: bool
    # Which feature format the finalization stage writes. The search stage is the
    # same for both; only what is built from its MSA bundles differs.
    backend: str = "alphafold3"
    binary_path: Path | None = None
    binary_id: str | None = None
    temp_dir: Path | None = None
    batch_max_sequences: int = 256
    batch_max_residues: int = 100_000
    e_value: float = 1e-4
    use_gpu: bool = True
    # Iterative profile search, as jackhmmer's n_iter. AlphaFold 3 uses 3; MMseqs2
    # defaults to 1, a plain sequence-sequence search. It changes which sequences
    # are found, so it belongs in the cache identity.
    num_iterations: int = 1
    # How MMseqs2 reads the target database (0 auto, 1 fread, 2 mmap, 3 mmap+touch).
    # Performance only: it never changes the alignment, so it is deliberately kept
    # out of the cache identity and switching it reuses existing bundles.
    db_load_mode: int | None = None
    search_ram_mb: int = 160_000
    gpu_ram_scaling: float = 1.1
    search_runtime_base_minutes: float = 90
    cpu_runtime_multiplier: float = 2.5
    finalize_base_ram_mb: int = 2_000
    finalize_ram_per_residue_mb: float = 1.0
    finalize_runtime_minutes_base: float = 30
    gpu_runtime_per_sequence_minutes: float = 0.5
    gpu_runtime_per_1000_residues: float = 1.0
    template_database_ids: Mapping[str, str] | None = None
    databases: Mapping[str, MmseqsDatabaseConfig] | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        data_pipeline: str,
    ) -> "LocalMmseqsFeatureConfig":
        values = dict(raw or {})
        enabled = _enabled(values.get("enabled", False))
        if not enabled:
            return cls(enabled=False)
        backend = _BACKENDS.get(str(data_pipeline).lower())
        if backend is None:
            raise ValueError(
                "Local MMseqs2 features need --data_pipeline alphafold2 or "
                f"alphafold3, not {data_pipeline!r}"
            )

        binary_path = Path(
            values.get("binary_path", _BUNDLED_MMSEQS_BINARY)
            or _BUNDLED_MMSEQS_BINARY
        )
        if binary_path != _BUNDLED_MMSEQS_BINARY:
            raise ValueError(
                "Local MMseqs2-GPU requires the bundled MMseqs2 binary at "
                f"{_BUNDLED_MMSEQS_BINARY}; host binaries are not visible in "
                "the prediction container"
            )
        binary_id = str(values.get("binary_id", _BUNDLED_MMSEQS_ID)).strip()
        if not binary_id:
            raise ValueError("Local MMseqs2-GPU requires a binary_id")

        database_values = values.get("databases", {})
        databases = {}
        for name in _DATABASE_NAMES:
            database = database_values.get(name)
            if not isinstance(database, Mapping):
                raise ValueError(
                    f"Local MMseqs2-GPU requires an explicit {name} database"
                )
            max_sequences = (
                int(database["max_sequences"])
                if database.get("max_sequences") is not None
                else None
            )
            if max_sequences is not None and max_sequences < 1:
                raise ValueError(
                    f"Local MMseqs2-GPU {name} max_sequences must be at least 1"
                )
            databases[name] = MmseqsDatabaseConfig(
                path=Path(_required(database, "path", f"{name} database")),
                identifier=str(
                    _required(database, "identifier", f"{name} database")
                ),
                max_sequences=max_sequences,
            )
        # RNA is opt-in and all-or-nothing: a partial set would silently shallow the
        # MSA, since AlphaFold 3 merges all three into one unpaired alignment.
        configured_rna = [
            name for name in _RNA_DATABASE_NAMES
            if isinstance(database_values.get(name), Mapping)
        ]
        if configured_rna and len(configured_rna) != len(_RNA_DATABASE_NAMES):
            missing = [n for n in _RNA_DATABASE_NAMES if n not in configured_rna]
            raise ValueError(
                "Local MMseqs2 RNA support needs all of "
                f"{', '.join(_RNA_DATABASE_NAMES)}; missing: {', '.join(missing)}"
            )
        for name in configured_rna:
            database = database_values[name]
            max_sequences = (
                int(database["max_sequences"])
                if database.get("max_sequences") is not None
                else None
            )
            if max_sequences is not None and max_sequences < 1:
                raise ValueError(
                    f"Local MMseqs2 {name} max_sequences must be at least 1"
                )
            databases[name] = MmseqsDatabaseConfig(
                path=Path(_required(database, "path", f"{name} database")),
                identifier=str(_required(database, "identifier", f"{name} database")),
                max_sequences=max_sequences,
            )
        batch_max_sequences = int(values.get("batch_max_sequences", 256))
        batch_max_residues = int(values.get("batch_max_residues", 100_000))
        e_value = float(values.get("e_value", 1e-4))
        use_gpu = _enabled(values.get("use_gpu", True))
        num_iterations = int(values.get("num_iterations", 1))
        if num_iterations < 1:
            raise ValueError(
                "Local MMseqs2 num_iterations must be at least 1"
            )
        raw_db_load_mode = values.get("db_load_mode")
        db_load_mode = (
            int(raw_db_load_mode) if raw_db_load_mode is not None else None
        )
        if db_load_mode is not None and db_load_mode not in (0, 1, 2, 3):
            raise ValueError(
                "Local MMseqs2 db_load_mode must be 0 (auto), 1 (fread), "
                "2 (mmap) or 3 (mmap+touch)"
            )
        search_ram_mb = int(values.get("search_ram_mb", 160_000))
        gpu_ram_scaling = float(values.get("gpu_ram_scaling", 1.1))
        search_runtime_base_minutes = float(
            values.get("search_runtime_base_minutes", 90)
        )
        cpu_runtime_multiplier = float(values.get("cpu_runtime_multiplier", 2.5))
        finalize_base_ram_mb = int(values.get("finalize_base_ram_mb", 2_000))
        finalize_ram_per_residue_mb = float(
            values.get("finalize_ram_per_residue_mb", 1.0)
        )
        finalize_runtime_minutes_base = float(
            values.get("finalize_runtime_minutes_base", 30)
        )
        gpu_runtime_per_sequence_minutes = float(
            values.get("gpu_runtime_per_sequence_minutes", 0.5)
        )
        gpu_runtime_per_1000_residues = float(
            values.get("gpu_runtime_per_1000_residues", 1.0)
        )
        for field, value in (
            ("batch_max_sequences", batch_max_sequences),
            ("batch_max_residues", batch_max_residues),
            ("e_value", e_value),
            ("search_ram_mb", search_ram_mb),
            ("gpu_ram_scaling", gpu_ram_scaling),
            ("search_runtime_base_minutes", search_runtime_base_minutes),
            ("cpu_runtime_multiplier", cpu_runtime_multiplier),
            ("finalize_base_ram_mb", finalize_base_ram_mb),
            ("finalize_ram_per_residue_mb", finalize_ram_per_residue_mb),
            ("finalize_runtime_minutes_base", finalize_runtime_minutes_base),
            ("gpu_runtime_per_sequence_minutes", gpu_runtime_per_sequence_minutes),
            ("gpu_runtime_per_1000_residues", gpu_runtime_per_1000_residues),
        ):
            if value <= 0:
                raise ValueError(
                    f"Local MMseqs2-GPU {field} must be greater than 0"
                )
        template_values = values.get("template_database_ids")
        if not isinstance(template_values, Mapping):
            raise ValueError(
                "Local MMseqs2-GPU requires template_database_ids"
            )
        template_database_ids = {
            name: str(
                _required(template_values, name, f"{name} template database")
            )
            for name in ("pdb_seqres", "mmcif")
        }

        return cls(
            enabled=True,
            backend=backend,
            binary_path=binary_path,
            binary_id=binary_id,
            temp_dir=Path(_required(values, "temp_dir", "configuration")),
            batch_max_sequences=batch_max_sequences,
            batch_max_residues=batch_max_residues,
            e_value=e_value,
            use_gpu=use_gpu,
            num_iterations=num_iterations,
            db_load_mode=db_load_mode,
            search_ram_mb=search_ram_mb,
            gpu_ram_scaling=gpu_ram_scaling,
            search_runtime_base_minutes=search_runtime_base_minutes,
            cpu_runtime_multiplier=cpu_runtime_multiplier,
            finalize_base_ram_mb=finalize_base_ram_mb,
            finalize_ram_per_residue_mb=finalize_ram_per_residue_mb,
            finalize_runtime_minutes_base=finalize_runtime_minutes_base,
            gpu_runtime_per_sequence_minutes=gpu_runtime_per_sequence_minutes,
            gpu_runtime_per_1000_residues=gpu_runtime_per_1000_residues,
            template_database_ids=template_database_ids,
            databases=databases,
        )

    @property
    def rna_configured(self) -> bool:
        """Whether RNA chains can be searched, i.e. all three databases are set."""
        configured = self.databases or {}
        return all(name in configured for name in _RNA_DATABASE_NAMES)

    def msa_cli_arguments(self, *, threads: int, memory_mb: int) -> tuple[str, ...]:
        """Arguments owned by the GPU MSA stage."""
        if not self.enabled or self.binary_path is None or self.temp_dir is None:
            return ()
        values = {
            "mmseqs_binary_path": self.binary_path,
            "mmseqs_temp_dir": self.temp_dir,
            "mmseqs_batch_max_sequences": self.batch_max_sequences,
            "mmseqs_batch_max_residues": self.batch_max_residues,
            "mmseqs_e_value": self.e_value,
            "mmseqs_use_gpu": "true" if self.use_gpu else "false",
            "mmseqs_threads": threads,
        }
        # Both are omitted at their defaults so an unchanged configuration still
        # produces the exact command line it produced before they existed.
        if self.num_iterations > 1:
            values["mmseqs_num_iterations"] = self.num_iterations
        if self.db_load_mode is not None:
            values["mmseqs_db_load_mode"] = self.db_load_mode
        # Without this MMseqs2 sizes its database splits from 90% of the PHYSICAL node
        # memory and ignores the cgroup, so on a large node with a small allocation it
        # declines to split and is OOM-killed. Tell it the allocation instead.
        #
        # Snakemake passes a TBDString here during a dry run, and on a real run before
        # the resource is resolved, so accept only a genuine number and otherwise leave
        # the option out - MMseqs2 then behaves as it did before.
        try:
            limit_mb = int(memory_mb)
        except (TypeError, ValueError):
            limit_mb = 0
        if limit_mb > 0:
            values["mmseqs_split_memory_limit"] = f"{max(int(limit_mb * 0.9), 1)}M"
        for name, database in (self.databases or {}).items():
            values[f"mmseqs_{name}_database_path"] = database.path
            values[f"mmseqs_{name}_database_id"] = database.identifier
            if database.max_sequences is not None:
                values[f"mmseqs_{name}_max_sequences"] = database.max_sequences
        return tuple(
            f"--{name}={shlex.quote(str(value))}" for name, value in values.items()
        )

    def template_arguments(
        self, create_feature_arguments: Mapping[str, Any] | None
    ) -> dict[str, str]:
        """The create_feature_arguments this backend's finalization reads.

        AlphaFold 3 runs its own native template search and takes none of them.
        AlphaFold 2 builds hmmsearch or hhsearch from them, so a user who pointed
        --pdb_seqres_database_path somewhere, or chose --use_hhsearch, gets the
        same template stack here as the native pipeline would have built.
        """
        if self.backend != "alphafold2":
            return {}
        arguments = dict(create_feature_arguments or {})
        return {
            name: str(arguments[name])
            for name in _AF2_TEMPLATE_ARGUMENTS
            if arguments.get(name) not in (None, "", False, "false", "False")
        }

    def finalize_cli_arguments(
        self, create_feature_arguments: Mapping[str, Any] | None = None
    ) -> tuple[str, ...]:
        """Arguments owned by the CPU finalization stage."""
        if not self.enabled:
            return ()
        identifiers = self.template_database_ids or {}
        return (
            f"--data_pipeline={self.backend}",
            "--template_seqres_database_id="
            + shlex.quote(identifiers["pdb_seqres"]),
            "--template_mmcif_database_id=" + shlex.quote(identifiers["mmcif"]),
            *(
                f"{name}={shlex.quote(value)}"
                for name, value in self.template_arguments(
                    create_feature_arguments
                ).items()
            ),
        )

    def _digest(self, values: Mapping[str, Any]) -> str:
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def msa_cache_key(self, prediction_container: str) -> str:
        """Namespace MSA caches by every workflow-visible search input.

        Every setting that changes which sequences the search finds has to appear
        here, because this key decides whether the workflow even RUNS the core
        stage. The core keys its own bundles on the same facts, but a completed
        shard summary short-circuits that check: flip a search setting the key
        ignores, and Snakemake sees the shard as done and never starts the job
        whose cache check would have caught it. That is what happened with
        ``use_gpu`` -- the core records cpu/gpu in its provenance, this key did
        not, so switching search mode silently kept the old alignments.

        Settings are recorded only when they differ from their default, so adding
        one does not orphan the caches of runs that never set it.
        """
        databases = {
            name: {
                "identifier": database.identifier,
                "max_sequences": database.max_sequences,
            }
            for name, database in sorted((self.databases or {}).items())
        }
        identity = {
            "container": prediction_container,
            "mmseqs_binary_id": self.binary_id,
            "e_value": self.e_value,
            "databases": databases,
        }
        if not self.use_gpu:
            identity["search_mode"] = "cpu"
        if self.num_iterations > 1:
            identity["num_iterations"] = self.num_iterations
        # db_load_mode is absent on purpose: see the field's comment.
        return self._digest(identity)

    def feature_cache_key(
        self,
        max_template_date: str,
        prediction_container: str,
        create_feature_arguments: Mapping[str, Any] | None = None,
    ) -> str:
        """Namespace final features by MSA, backend and template provenance.

        The backend and AlphaFold 2's template settings are recorded only when they
        apply, so every AlphaFold 3 feature cache keeps the namespace it had.
        """
        identity = {
            "msa": self.msa_cache_key(prediction_container),
            "max_template_date": str(max_template_date),
            "template_database_ids": dict(self.template_database_ids or {}),
        }
        if self.backend != "alphafold3":
            identity["backend"] = self.backend
            identity["template_arguments"] = self.template_arguments(
                create_feature_arguments
            )
        return self._digest(identity)

    def search_memory_mb(self, *, safety: float, attempt: int, cap_mb: int = 0) -> int:
        """Host RAM for one search shard, sized from the largest configured database.

        Measured flat in shard size (1, 8, 32 and 128 queries all peaked identically)
        and flat in query length (1144 and 8268 residues likewise), because the cost is
        the target database, not the queries. MMseqs2's own estimator takes no query
        argument at all. So there is no per-residue or per-query term here.

        Databases are searched one after another, so the peak tracks the LARGEST
        database rather than their total: measured 128.7 GB for mgnify alone against
        149.4 GB for all four, where the sum of the individual peaks was 283.2 GB.
        Peak RSS ran 0.69-0.85x the padded size on disk, so when the configured
        databases are readable the estimate is derived from them and search_ram_mb is
        only the fallback for when they are not.
        """
        derived = self._largest_database_mb()
        estimate = safety * (derived or self.search_ram_mb)
        # A nucleotide search does not follow the padded-protein disk ratio, so the
        # derivation above under-requests it by roughly 3.6x. Floor it at the measured
        # boundary whenever an RNA database is configured, since any shard may contain
        # an RNA chain.
        if self.rna_configured:
            estimate = max(estimate, safety * _RNA_SEARCH_FLOOR_MB)
        value = math.ceil(
            estimate * (self.gpu_ram_scaling ** max(int(attempt) - 1, 0))
        )
        return min(value, cap_mb) if cap_mb else value

    def finalize_memory_mb(
        self, residues: int, *, safety: float, attempt: int, cap_mb: int = 0
    ) -> int:
        """Host RAM for finalizing one protein: template search plus artifact writing.

        Measured 0.24 GB at 117 residues and 0.57 GB at 887 (shard of eight), i.e. a
        small constant with a shallow slope - two orders of magnitude below the
        MSA-generation model this stage used to borrow.
        """
        estimate = safety * (
            self.finalize_base_ram_mb
            + self.finalize_ram_per_residue_mb * max(int(residues), 0)
        )
        value = math.ceil(estimate * (self.gpu_ram_scaling ** max(int(attempt) - 1, 0)))
        return min(value, cap_mb) if cap_mb else value

    def finalize_runtime_minutes(self, *, attempt: int) -> int:
        """Wall time for one finalization. Measured ~14 s per protein."""
        return math.ceil(self.finalize_runtime_minutes_base * max(int(attempt), 1))

    def _largest_database_mb(self) -> int:
        """Expected peak from the largest configured database, or 0 if unreadable."""
        largest = 0
        for database in (self.databases or {}).values():
            total = 0
            for path in glob.glob(f"{database.path}*"):
                try:
                    total += os.path.getsize(path)
                except OSError:
                    return 0
            largest = max(largest, total)
        if not largest:
            return 0
        # 0.85 covers the highest measured RSS/disk ratio; 1.16 is the observed
        # overhead of chaining several databases in one process.
        return math.ceil(largest / 1024**2 * 0.85 * 1.16)

    def search_runtime_minutes(
        self, sequences: int, residues: int, *, attempt: int
    ) -> int:
        """Wall time for one search shard.

        Also dominated by the database scan: 1 query took 65 min and 128 took 66 min on
        one L40S, so the base term carries the estimate and the per-query terms are a
        small margin. CPU search measured 2.4x slower than GPU on the same shard.
        """
        base = self.search_runtime_base_minutes
        if not self.use_gpu:
            base *= self.cpu_runtime_multiplier
        estimate = (
            base
            + self.gpu_runtime_per_sequence_minutes * max(int(sequences), 0)
            + self.gpu_runtime_per_1000_residues
            * math.ceil(max(int(residues), 0) / 1000)
        )
        return math.ceil(estimate * max(int(attempt), 1))

    @property
    def bind_paths(self) -> tuple[Path, ...]:
        """Host paths that must be visible inside the prediction container."""
        if not self.enabled or self.binary_path is None or self.temp_dir is None:
            return ()

        def parent_and_target(path: Path) -> tuple[Path, ...]:
            candidates = [path.parent]
            try:
                candidates.append(path.resolve().parent)
            except (OSError, RuntimeError):
                pass
            return tuple(candidates)

        candidates = [self.temp_dir]
        try:
            candidates.append(self.temp_dir.resolve())
        except (OSError, RuntimeError):
            pass
        for database in (self.databases or {}).values():
            candidates.extend(parent_and_target(database.path))
        return tuple(dict.fromkeys(candidates))
