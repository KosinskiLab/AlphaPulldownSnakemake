""" Snakemake I/O and utility functions

    Copyright (c) 2024 European Molecular Biology Laboratory

    Authors: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable


def feature_suffix(compression: str = "lzma") -> str:
    _compression = {
        "lzma": "xz",
    }
    suffix = _compression.get(compression, None)
    ret = "pkl"
    if suffix is not None:
        ret += f".{suffix}"
    return ret


def _first_level_root(p: Path) -> Path | None:
    try:
        p = p.expanduser()
        if not p.is_absolute():
            p = p.resolve()
        parts = p.parts
        if len(parts) >= 2:
            root = Path("/" + parts[1])
            if root.exists():
                return root
    except (OSError, RuntimeError):
        pass
    return None


def _collect_roots(paths: Iterable[str | Path]) -> set[str]:
    roots: set[str] = set()
    for raw in paths:
        try:
            p = Path(raw)
            r1 = _first_level_root(p)
            if r1:
                roots.add(str(r1))
            try:
                rp = p.expanduser().resolve()
                r2 = _first_level_root(rp)
                if r2:
                    roots.add(str(r2))
            except (OSError, RuntimeError):
                pass
        except (TypeError, OSError, RuntimeError):
            pass
    return roots


def prepare_container_binds(
    *,
    output_directory: str,
    config: dict[str, Any],
    feature_directories: Iterable[str | Path] = (),
    input_files: Iterable[str | Path] = (),
) -> None:
    """Populate Singularity/Apptainer bind paths based on config."""
    interest: set[Path] = {
        Path(__file__).parent,
        Path.cwd(),
        Path(output_directory),
    }

    for key in ("databases_directory", "backend_weights_directory", "features_directory"):
        value = config.get(key)
        if value:
            interest.add(Path(value))

    for path in feature_directories:
        interest.add(Path(path))

    for path in input_files:
        try:
            interest.add(Path(path).expanduser().resolve().parent)
        except (TypeError, OSError, RuntimeError):
            continue

    roots = sorted(_collect_roots(interest))
    bind_spec = ",".join(f"{r}:{r}" for r in roots)

    for var in ("APPTAINER_BINDPATH", "SINGULARITY_BINDPATH"):
        os.environ.setdefault(var, bind_spec)
    for var in ("APPTAINER_NV", "SINGULARITY_NV"):
        os.environ.setdefault(var, "1")


def linear_resources(
    *,
    mem: int = 800,
    runtime: int = 10,
    avg_factor: float = 0.75,
    mem_fn: Callable[[Any, int], float] | None = None,
    runtime_fn: Callable[[Any, int], float] | None = None,
    attempt_fn: Callable[[Any, int], int] | None = None,
) -> dict[str, Any]:
    """Return a Snakemake resources dictionary scaling with retry attempts."""

    def _mem_value(wc, attempt: int) -> float:
        if mem_fn:
            return float(mem_fn(wc, attempt))
        return float(mem * attempt)

    def _runtime_value(wc, attempt: int) -> float:
        if runtime_fn:
            return float(runtime_fn(wc, attempt))
        return float(runtime * attempt)

    def _avg_mem(wc, attempt: int) -> int:
        return int(_mem_value(wc, attempt) * avg_factor)

    def _mem_mb(wc, attempt: int) -> int:
        return int(_mem_value(wc, attempt))

    def _runtime(wc, attempt: int) -> int:
        return int(_runtime_value(wc, attempt))

    def _attempt(wc, attempt: int) -> int:
        if attempt_fn:
            return int(attempt_fn(wc, attempt))
        return attempt

    return {
        "avg_mem": _avg_mem,
        "mem_mb": _mem_mb,
        "runtime": _runtime,
        "attempt": _attempt,
    }
