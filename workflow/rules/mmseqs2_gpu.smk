"""Thin Snakemake adapter for batched local MMseqs2-GPU feature generation.

The AlphaPulldown package owns feature generation.  This module translates
workflow configuration and exposes required bind paths without exposing
process or chunking details to the Snakefile.
"""

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any, Mapping


_DATABASE_NAMES = ("uniref90", "mgnify", "small_bfd", "uniprot")
_BUNDLED_MMSEQS_BINARY = Path("/opt/mmseqs/bin/mmseqs")


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
class LocalMmseqsFeatureConfig:
    enabled: bool
    binary_path: Path | None = None
    temp_dir: Path | None = None
    batch_max_sequences: int = 256
    batch_max_residues: int = 100_000
    sensitivity: float = 7.5
    e_value: float = 1e-4
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
        if data_pipeline.lower() not in {"alphafold3", "af3"}:
            raise ValueError("Local MMseqs2-GPU features require AlphaFold 3")

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
        batch_max_sequences = int(values.get("batch_max_sequences", 256))
        batch_max_residues = int(values.get("batch_max_residues", 100_000))
        sensitivity = float(values.get("sensitivity", 7.5))
        e_value = float(values.get("e_value", 1e-4))
        for field, value in (
            ("batch_max_sequences", batch_max_sequences),
            ("batch_max_residues", batch_max_residues),
            ("sensitivity", sensitivity),
            ("e_value", e_value),
        ):
            if value <= 0:
                raise ValueError(
                    f"Local MMseqs2-GPU {field} must be greater than 0"
                )
        return cls(
            enabled=True,
            binary_path=Path(
                values.get("binary_path", _BUNDLED_MMSEQS_BINARY)
                or _BUNDLED_MMSEQS_BINARY
            ),
            temp_dir=Path(_required(values, "temp_dir", "configuration")),
            batch_max_sequences=batch_max_sequences,
            batch_max_residues=batch_max_residues,
            sensitivity=sensitivity,
            e_value=e_value,
            databases=databases,
        )

    def cli_arguments(self, *, threads: int) -> tuple[str, ...]:
        """Translate the workflow configuration to the core CLI interface."""
        if not self.enabled or self.binary_path is None or self.temp_dir is None:
            return ()
        values = {
            "mmseqs_binary_path": self.binary_path,
            "mmseqs_temp_dir": self.temp_dir,
            "mmseqs_batch_max_sequences": self.batch_max_sequences,
            "mmseqs_batch_max_residues": self.batch_max_residues,
            "mmseqs_sensitivity": self.sensitivity,
            "mmseqs_e_value": self.e_value,
            "mmseqs_threads": threads,
        }
        for name, database in (self.databases or {}).items():
            values[f"mmseqs_{name}_database_path"] = database.path
            values[f"mmseqs_{name}_database_id"] = database.identifier
            if database.max_sequences is not None:
                values[f"mmseqs_{name}_max_sequences"] = database.max_sequences
        return tuple(
            f"--{name}={shlex.quote(str(value))}" for name, value in values.items()
        )

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
