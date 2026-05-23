""" Snakemake I/O and utility functions

    Copyright (c) 2024 European Molecular Biology Laboratory

    Authors: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable


@functools.lru_cache(maxsize=None)
def residue_count(fasta_path: str) -> int:
    """Number of residues in a (single-record) FASTA file.

    Counts sequence characters, ignoring the header line(s) and whitespace.
    Returns 0 when the file cannot be read yet (e.g. during a dry-run before
    the upstream download/symlink rule has produced it) so that resource
    estimation degrades gracefully to the base allocation instead of crashing.
    Results are memoised because the structure-inference estimator may look up
    the same chain repeatedly within a workflow.
    """
    try:
        total = 0
        with open(fasta_path) as handle:
            for line in handle:
                if line.startswith(">"):
                    continue
                total += len(line.strip())
        return total
    except OSError:
        return 0


def fold_total_tokens(fold: str, data_dir: str, delimiter: str = "+") -> int:
    """Total residue (token) count of a fold specification.

    Sums the residue length of every chain in ``fold``, honouring copy numbers
    such as ``A:2`` (a homo-dimer counts twice). Region selections such as
    ``A:1-100`` are conservatively counted at the chain's full length, which
    over- rather than under-estimates memory. Per-chain lengths are read from
    ``<data_dir>/<chain>.fasta``.
    """
    total = 0
    for token in str(fold).split(delimiter):
        parts = [part for part in token.split(":") if part]
        if not parts:
            continue
        name = parts[0]
        copies = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 1
        total += residue_count(os.path.join(data_dir, f"{name}.fasta")) * copies
    return total


def _cap_mem(value_mb: float, cap_mb: int) -> int:
    value = max(int(value_mb), 1)
    if cap_mb and cap_mb > 0:
        value = min(value, int(cap_mb))
    return value


def estimate_feature_mem_mb(
    seq_len: int,
    *,
    base_mb: float,
    per_residue_mb: float,
    scaling: float,
    safety: float,
    attempt: int,
    cap_mb: int = 0,
) -> int:
    """Length-aware host RAM (MB) for the feature-generation (MSA) stage.

    Feature memory is dominated by a near-fixed database/MSA-tooling footprint
    with only a mild dependence on query length, so the model is linear:

        mem = safety * (base_mb + per_residue_mb * seq_len)

    The first attempt already carries the ``safety`` margin; OOM retries
    escalate further via ``scaling ** (attempt - 1)``.
    """
    estimate = safety * (base_mb + per_residue_mb * max(int(seq_len), 0))
    value = estimate * (scaling ** max(int(attempt) - 1, 0))
    return _cap_mem(value, cap_mb)


def estimate_inference_mem_mb(
    total_tokens: int,
    *,
    base_mb: float,
    per_token_sq_mb: float,
    scaling: float,
    safety: float,
    attempt: int,
    cap_mb: int = 0,
) -> int:
    """Length-aware host RAM (MB) for the structure-inference stage.

    AlphaFold's pair representation is O(N^2) in the number of tokens N (total
    residues of the complex), so peak memory follows a quadratic:

        mem = safety * (base_mb + per_token_sq_mb * N**2)

    With unified memory enabled the XLA fraction is derived from this host
    allocation, so sizing host RAM by N also sizes the GPU spill ceiling. The
    first attempt carries the ``safety`` margin; OOM retries escalate via
    ``scaling ** (attempt - 1)``.
    """
    estimate = safety * (base_mb + per_token_sq_mb * (max(int(total_tokens), 0) ** 2))
    value = estimate * (scaling ** max(int(attempt) - 1, 0))
    return _cap_mem(value, cap_mb)


# Backend-specific memory defaults. AlphaFold-Multimer (AF2) inference carries a
# substantially heavier host-RAM footprint than AlphaFold 3 at the same complex
# size (measured host RSS ~4x higher around N~2300 in the benchmark campaign), and
# its feature stage runs HHblits, the dominant OOM source; the AF3 data pipeline
# (jackhmmer/nhmmer, no HHblits) is lighter. So AF2 gets larger bases and a larger
# quadratic term. These apply only when the matching config key is unset, so an
# explicit config value always wins.
FEATURE_RAM_DEFAULTS = {
    "alphafold2": {"base_mb": 64000, "per_residue_mb": 40},
    "alphafold3": {"base_mb": 40000, "per_residue_mb": 25},
}
INFERENCE_RAM_DEFAULTS = {
    # base_mb: fixed floor; per_token_sq_mb: quadratic coeff in N^2 (total residues).
    # AF2 base/coeff cover the measured AF2 host RSS with margin; the AF3 quadratic is
    # sized to the observed GPU-VRAM demand so the unified-memory spill ceiling
    # (host_mem / gpu_vram) covers large complexes instead of OOM-ing.
    "alphafold2": {"base_mb": 24000, "per_token_sq_mb": 0.0055, "runtime_minutes": 1440},
    "alphafold3": {"base_mb": 16000, "per_token_sq_mb": 0.0045, "runtime_minutes": 1440},
}


def normalize_backend(name, default: str = "alphafold2") -> str:
    """Map a backend/data-pipeline string to 'alphafold2' or 'alphafold3'."""
    n = str(name if name is not None else default).strip().lower()
    return "alphafold3" if n in ("alphafold3", "af3") else "alphafold2"


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
    """Return a Snakemake resources dictionary scaling with retry attempts.

    User-supplied ``*_fn`` callbacks receive ``wildcards`` positionally and may
    additionally declare ``input`` and/or ``attempt`` parameters; only the ones
    they declare are forwarded. This keeps legacy ``f(wc, attempt)`` callbacks
    working while letting length-aware callbacks read input files via
    ``f(wildcards, input, attempt)``.
    """

    def _invoke(fn, wc, input, attempt):
        params = inspect.signature(fn).parameters
        kwargs = {}
        if "input" in params:
            kwargs["input"] = input
        if "attempt" in params:
            kwargs["attempt"] = attempt
        return fn(wc, **kwargs)

    def _mem_value(wc, input, attempt: int) -> float:
        if mem_fn:
            return float(_invoke(mem_fn, wc, input, attempt))
        return float(mem * attempt)

    def _runtime_value(wc, input, attempt: int) -> float:
        if runtime_fn:
            return float(_invoke(runtime_fn, wc, input, attempt))
        return float(runtime * attempt)

    def _avg_mem(wc, input, attempt: int) -> int:
        return int(_mem_value(wc, input, attempt) * avg_factor)

    def _mem_mb(wc, input, attempt: int) -> int:
        return int(_mem_value(wc, input, attempt))

    def _runtime(wc, input, attempt: int) -> int:
        return int(_runtime_value(wc, input, attempt))

    def _attempt(wc, input, attempt: int) -> int:
        if attempt_fn:
            return int(_invoke(attempt_fn, wc, input, attempt))
        return attempt

    return {
        "avg_mem": _avg_mem,
        "mem_mb": _mem_mb,
        "runtime": _runtime,
        "attempt": _attempt,
    }
