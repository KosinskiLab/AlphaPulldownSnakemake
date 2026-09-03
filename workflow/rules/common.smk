""" Snakemake I/O and utility functions

    Copyright (c) 2024 European Molecular Biology Laboratory

    Authors: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

# Default maximum *total* complex length (residues) per backend, used to skip
# folds that are too large to be feasible. AF3 supports larger inputs than
# AF2-Multimer. Override via config (see Snakefile / config.yaml).
MAX_TOTAL_LENGTH_DEFAULTS = {"alphafold2": 5000, "alphafold3": 7000}


# Length lookups cache only *successful* (>0) reads. Caching a 0 from a not-yet-
# created file would be a correctness bug: Snakemake's scheduler evaluates resource
# functions early (before upstream download/symlink rules run), and a memoised 0
# would then stick even after the file appears, collapsing length-aware sizing to
# the base allocation. Re-reading a small FASTA/JSON on later calls is cheap.
_RESIDUE_COUNT_CACHE: dict[str, int] = {}
_AF3_INPUT_COUNT_CACHE: dict[str, int] = {}


def residue_count(fasta_path: str) -> int:
    """Number of residues in a (single-record) FASTA file.

    Counts sequence characters, ignoring header lines and whitespace. Returns 0
    when the file cannot be read yet (so estimation degrades to the base
    allocation rather than crashing) and does not cache that 0 — see the note
    above on why caching a missing-file result would be wrong.
    """
    cached = _RESIDUE_COUNT_CACHE.get(fasta_path)
    if cached:
        return cached
    try:
        total = 0
        with open(fasta_path) as handle:
            for line in handle:
                if not line.startswith(">"):
                    total += len(line.strip())
    except OSError:
        return 0
    if total > 0:
        _RESIDUE_COUNT_CACHE[fasta_path] = total
    return total


def af3_input_residue_count(json_path: str) -> int:
    """Total polymer residues in an AlphaFold 3 ``*_af3_input.json`` feature file.

    Sums the ``sequence`` length of every protein/RNA/DNA entry under
    ``sequences`` (ligands have no sequence and are skipped). Returns 0 if the
    file is missing or not parseable (not cached); used as a fallback for the
    chain length when no ``data/<chain>.fasta`` exists (e.g. precomputed features
    supplied via ``feature_directory``).
    """
    cached = _AF3_INPUT_COUNT_CACHE.get(json_path)
    if cached:
        return cached
    try:
        with open(json_path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return 0
    total = 0
    for entry in data.get("sequences", []):
        if not isinstance(entry, dict):
            continue
        for mol in ("protein", "rna", "dna"):
            mol_entry = entry.get(mol)
            if isinstance(mol_entry, dict):
                total += len(mol_entry.get("sequence", "") or "")
    if total > 0:
        _AF3_INPUT_COUNT_CACHE[json_path] = total
    return total


# Re-export the canonical fold-spec parser from `alphapulldown-input-parser` under a
# short local name. Adapts the parser's `(name, copies, regions)` triples to the
# `(name, copies)` pairs the memory/length-filter logic here uses (regions are
# conservatively counted at full chain length — see `fold_total_tokens` for why).
from alphapulldown_input_parser import (
    parse_fold_chains as _parse_fold_chains_with_regions,  # noqa: E402
)


def parse_fold_chains(fold: str, delimiter: str = "+") -> list[tuple[str, int]]:
    """Delegate to ``alphapulldown_input_parser.parse_fold_chains``; drop regions."""
    return [
        (name, copies)
        for name, copies, _regions in _parse_fold_chains_with_regions(fold, delimiter)
    ]


def is_json_input(name: str) -> bool:
    """True if a fold token names a direct AF3 JSON input (e.g. a ``ligand.json``).

    Such tokens are AlphaFold 3 inputs supplied as-is via ``feature_directory``;
    they are *not* proteins and must never be downloaded or sent through feature
    generation. Everything else is treated as a protein chain reference.
    """
    return str(name).lower().endswith(".json")


def split_fold_inputs(
    fold: str, delimiter: str = "+"
) -> tuple[list[str], list[str]]:
    """Partition a fold spec into protein chains and direct AF3 JSON inputs.

    Returns ``(protein_bases, json_basenames)``:

    - ``protein_bases``  -- base names (path + extension stripped) of chains that
      need feature generation/download, mirroring the parser's stem handling.
    - ``json_basenames`` -- basenames of ``*.json`` tokens supplied directly as AF3
      inputs (e.g. ligands), which are provided via ``feature_directory`` and never
      generated.

    Both lists preserve first-seen order and are de-duplicated. Copy numbers and
    region ranges (``ligand.json:80``, ``A:1-100``) are stripped by the underlying
    chain parser, so only the chain name survives here.
    """
    protein_bases: list[str] = []
    json_basenames: list[str] = []
    seen_proteins: set[str] = set()
    seen_json: set[str] = set()
    for name, _copies in parse_fold_chains(fold, delimiter):
        if is_json_input(name):
            base = os.path.basename(name)
            if base not in seen_json:
                seen_json.add(base)
                json_basenames.append(base)
        else:
            base = os.path.splitext(os.path.basename(name))[0]
            if base not in seen_proteins:
                seen_proteins.add(base)
                protein_bases.append(base)
    return protein_bases, json_basenames


def format_af3_requested_fold(fold: str, delimiter: str = "+") -> str:
    """Convert a logical fold spec into AlphaFold 3 inference ``--input`` tokens.

    Protein chains map to their generated feature file ``<base>_af3_input.json``;
    tokens that are already ``*.json`` (direct AF3 JSON inputs such as ligands) are
    passed through unchanged. Copy numbers and region ranges are preserved after the
    file name.

    Examples:
        ``P01258+P0AEZ3:2``       -> ``P01258_af3_input.json+P0AEZ3_af3_input.json:2``
        ``P01258+ligand.json:80`` -> ``P01258_af3_input.json+ligand.json:80``
        ``P01258:1-100:2``        -> ``P01258_af3_input.json:1-100:2``

    Rationale:
        - Protein features are generated as ``<base>_af3_input.json``.
        - JSON inputs are already AF3 inputs and must not get a second suffix.
        - Copy numbers / region ranges apply to the logical chain, not the file
          name; ``alphapulldown-input-parser`` accepts them after the JSON filename.
    """
    converted_parts: list[str] = []
    for token in str(fold).split(delimiter):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(":") if p.strip()]
        base = parts[0]
        suffix = ":".join(parts[1:]) if len(parts) > 1 else ""
        json_name = base if is_json_input(base) else f"{base}_af3_input.json"
        converted_parts.append(f"{json_name}:{suffix}" if suffix else json_name)
    return delimiter.join(converted_parts)


@functools.lru_cache(maxsize=None)
def fetch_uniprot_length(uniprot_id: str, timeout: float = 30.0) -> int:
    """Residue length of a UniProt entry via the REST API; 0 on any failure.

    Mirrors the reference snippet in issue #33. Used at parse time for length
    filtering when no local FASTA is available yet; failures return 0 so the
    caller can fail open (keep the fold) rather than crash offline.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except Exception:
        return 0
    total = 0
    for line in text.splitlines():
        if not line.startswith(">"):
            total += len(line.strip())
    return total


def chain_residue_count(
    name: str,
    data_dir: str,
    features_dir: str | None = None,
    is_af3: bool = False,
    length_cache: dict | None = None,
) -> int:
    """Residue length of a single chain.

    Resolution order: ``<data_dir>/<name>.fasta`` -> AF3 precomputed
    ``<features_dir>/<name>_af3_input.json`` (AF3 only) -> ``length_cache`` (the
    parse-time length table, which covers the AF2 precomputed-feature case where
    neither a FASTA nor an AF3 JSON exists). Returns 0 when length is unknown so
    sizing degrades to the base allocation plus retry escalation.

    A direct AF3 JSON input (``ligand.json``) is read from the file itself in
    ``features_dir``; ligand-only inputs have no polymer ``sequence`` and so
    contribute 0 (consistent with AF3 ligand atoms not being counted as tokens).
    """
    if is_json_input(name):
        if features_dir:
            return af3_input_residue_count(
                os.path.join(features_dir, os.path.basename(name))
            )
        return 0
    length = residue_count(os.path.join(data_dir, f"{name}.fasta"))
    if length == 0 and is_af3 and features_dir:
        length = af3_input_residue_count(
            os.path.join(features_dir, f"{name}_af3_input.json")
        )
    if length == 0 and length_cache:
        length = int(length_cache.get(name, 0) or 0)
    return length


def fold_total_tokens(
    fold: str,
    data_dir: str,
    delimiter: str = "+",
    features_dir: str | None = None,
    is_af3: bool = False,
    length_cache: dict | None = None,
) -> int:
    """Total residue (token) count of a fold specification.

    Sums the residue length of every chain in ``fold``, honouring copy numbers
    such as ``A:2`` (a homo-dimer counts twice). Region selections such as
    ``A:1-100`` are conservatively counted at the chain's full length, which
    over- rather than under-estimates memory. Per-chain lengths come from
    ``chain_residue_count`` (FASTA -> AF3 JSON -> length cache).

    Note: AF3 ligand atoms are not counted (no ``sequence`` field); for
    protein/nucleic complexes this matches the token count, and the safety
    margin plus retry escalation absorb any small ligand undercount.
    """
    total = 0
    for name, copies in parse_fold_chains(fold, delimiter):
        total += (
            chain_residue_count(name, data_dir, features_dir, is_af3, length_cache)
            * copies
        )
    return total


def fold_length_violation(
    chain_lengths: list[tuple[str, int | None, int]],
    max_protein_length: int = 0,
    max_total_length: int = 0,
) -> str | None:
    """Return a human-readable reason if a fold exceeds a length limit, else None.

    ``chain_lengths`` is a list of ``(name, length_or_None, copies)``. Limits of
    0 (or negative) are disabled. Unknown lengths (``None``) are treated as 0 so
    the decision fails open (the fold is kept) rather than dropped on missing data.
    """
    if max_protein_length and max_protein_length > 0:
        for name, length, _copies in chain_lengths:
            if length is not None and length > max_protein_length:
                return (
                    f"protein {name} length {length} exceeds "
                    f"max_protein_length {max_protein_length}"
                )
    if max_total_length and max_total_length > 0:
        total = sum((length or 0) * copies for _name, length, copies in chain_lengths)
        if total > max_total_length:
            return f"total length {total} exceeds max_total_length {max_total_length}"
    return None


def required_gpu_vram_gb(
    total_tokens: int, per_token_sq_mb: float, headroom: float = 1.0
) -> float:
    """Estimated peak GPU VRAM (GB) for a complex of ``total_tokens``.

    Uses the same O(N^2) coefficient as host-memory sizing as a proxy for on-device
    peak demand, scaled by ``headroom`` (e.g. 0.8 tolerates ~20% spill to host).
    """
    return headroom * per_token_sq_mb * (max(int(total_tokens), 0) ** 2) / 1000.0


def gpu_exclude_nodes(
    total_tokens: int,
    tiers,
    per_token_sq_mb: float,
    headroom: float = 1.0,
    extra_exclude: str = "",
) -> str:
    """Comma-joined SLURM nodes to exclude so a complex lands on a big-enough GPU.

    ``tiers`` is an iterable of ``{"min_vram_gb": int, "nodes": "<slurm hostlist>"}``
    describing the cluster's GPU pool. The complex's required VRAM
    (:func:`required_gpu_vram_gb`) selects the smallest tier that satisfies it (the
    largest tier if none does — the remainder spills to host via unified memory);
    the nodes of every *smaller* tier are excluded, so the job may run on any GPU at
    or above the chosen tier (the whole pool, not one pinned model). ``extra_exclude``
    (the static ``slurm_exclude_nodes``) is always appended.

    Cluster-agnostic: each site lists its own GPU tiers/nodes; nothing about a
    specific cluster is hard-coded.
    """
    parts: list[str] = []
    valid = [t for t in tiers if t and t.get("nodes")]
    if valid:
        ordered = sorted(valid, key=lambda t: int(t["min_vram_gb"]))
        required = required_gpu_vram_gb(total_tokens, per_token_sq_mb, headroom)
        chosen = len(ordered) - 1
        for index, tier in enumerate(ordered):
            if int(tier["min_vram_gb"]) >= required:
                chosen = index
                break
        parts.extend(str(tier["nodes"]) for tier in ordered[:chosen])
    if extra_exclude:
        parts.append(str(extra_exclude))
    return ",".join(part for part in parts if part)


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
# AF3 sized from measured peak RSS of 19k create_features jobs (median 1.1 GB,
# p99 17, max 35): 40000/25 asked ~66 GB for that. 24000/8 covers all of them.
FEATURE_RAM_DEFAULTS = {
    "alphafold2": {"base_mb": 64000, "per_residue_mb": 120},
    "alphafold3": {"base_mb": 24000, "per_residue_mb": 8},
}
INFERENCE_RAM_DEFAULTS = {
    # base_mb: fixed floor; per_token_sq_mb: quadratic coeff in N^2 (total residues).
    # Bases are deliberately modest: at low N the safety factor (1.25 by default) keeps a
    # comfortable margin over measured host RSS (~6 GB AF3 / ~6-30 GB AF2), and at high N
    # the quadratic dominates anyway. Retries still escalate via `scaling ** (attempt-1)`,
    # so an under-provisioned job self-heals (mem grows on each retry from this base).
    "alphafold2": {"base_mb": 16000, "per_token_sq_mb": 0.0055, "runtime_minutes": 1440},
    "alphafold3": {"base_mb":  8000, "per_token_sq_mb": 0.0045, "runtime_minutes": 1440},
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
    extra_paths: Iterable[str | Path] = (),
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

    for path in extra_paths:
        try:
            interest.add(Path(path))
        except (TypeError, OSError, RuntimeError):
            continue

    roots = sorted(_collect_roots(interest))
    bind_spec = ",".join(f"{r}:{r}" for r in roots)

    for var in ("APPTAINER_BINDPATH", "SINGULARITY_BINDPATH"):
        existing = [item for item in os.environ.get(var, "").split(",") if item]
        required = [item for item in bind_spec.split(",") if item]
        os.environ[var] = ",".join(dict.fromkeys([*existing, *required]))
    for var in ("APPTAINER_NV", "SINGULARITY_NV"):
        os.environ.setdefault(var, "1")


def batch_inference_args(
    base_args: dict,
    *,
    backend: str,
    batch_size: int,
    jax_cache_dir: str,
) -> dict:
    """Return inference CLI args with the batch-only flags added, each gated to the
    backend that accepts them.

    ``run_structure_prediction.py`` validates its flags per backend and hard-errors on
    any it does not recognise (``ValueError: not supported by backend '<name>'``), so a
    batch-only flag must ONLY be added for the backend(s) that accept it:

    * ``--allow_resume`` (AlphaFold2 only): a crashed batch re-runs all its folds, so
      resume the ones already done. AlphaFold3 rejects it.
    * ``--jax_compilation_cache_dir`` (AlphaFold3 only): lets the per-fold calls in a
      batch share one on-disk JAX compile cache. This is a JAX/XLA flag; AlphaFold2
      rejects it (its inference is not JAX-compiled).

    With ``batch_size <= 1`` nothing is added (the unbatched pipeline is untouched). Any
    value the user already set is preserved (``setdefault``).
    """
    args = dict(base_args)
    if batch_size > 1:
        if backend == "alphafold2":
            args.setdefault("--allow_resume", "true")
        if backend == "alphafold3":
            args.setdefault("--jax_compilation_cache_dir", jax_cache_dir)
    return args


# Inference flags each backend accepts, mirroring run_structure_prediction.py's
# ``_validate_flags_for_backend``. Names are WITHOUT the leading ``--``. This is only
# used for a parse-time WARNING: the container is the source of truth and hard-errors,
# so if this list drifts (a newer image adds a flag) the worst case is a spurious
# warning, never a blocked run. Keep in sync with AlphaPulldown when convenient.
_COMMON_INFERENCE_FLAGS = {
    "input", "output_directory", "data_directory", "features_directory",
    "protein_delimiter", "fold_backend", "random_seed", "storage_mode",
}
_AF2_LIKE_INFERENCE_FLAGS = {
    "compress_result_pickles", "remove_result_pickles", "models_to_relax",
    "relax_best_score_threshold", "remove_keys_from_pickles", "convert_to_modelcif",
    "allow_resume", "num_cycle", "num_predictions_per_model", "pair_msa",
    "save_features_for_multimeric_object", "skip_templates", "msa_depth_scan",
    "multimeric_template", "model_names", "msa_depth", "description_file",
    "path_to_mmt", "threshold_clashes", "hb_allowance", "plddt_threshold",
    "desired_num_res", "desired_num_msa", "benchmark", "model_preset",
    "use_ap_style", "use_gpu_relax", "dropout",
}
_AF3_INFERENCE_FLAGS = {
    "jax_compilation_cache_dir", "buckets", "flash_attention_implementation",
    "num_diffusion_samples", "num_seeds", "debug_templates", "debug_msas",
    "num_recycles", "save_embeddings", "save_distogram", "use_ap_style",
}
_ALPHALINK_EXTRA_FLAGS = {"crosslinks"}

ALLOWED_INFERENCE_FLAGS = {
    "alphafold2": _COMMON_INFERENCE_FLAGS | _AF2_LIKE_INFERENCE_FLAGS,
    "alphalink": _COMMON_INFERENCE_FLAGS | _AF2_LIKE_INFERENCE_FLAGS | _ALPHALINK_EXTRA_FLAGS,
    "alphafold3": _COMMON_INFERENCE_FLAGS | _AF3_INFERENCE_FLAGS,
}


def unknown_inference_flags(args, backend: str) -> list:
    """Return the ``structure_inference_arguments`` keys the given backend does not
    accept (leading ``--`` and any ``=value`` ignored), preserving input order.

    ``run_structure_prediction.py`` aborts the inference job on the first flag outside
    its per-backend allow set (``ValueError: not supported by backend '<name>'``), deep
    inside a Slurm job. Calling this at parse time lets the workflow warn on the head
    node in seconds instead. Returns ``[]`` when the backend name is unrecognised (we
    cannot judge, so stay silent) or every flag is accepted.
    """
    allowed = ALLOWED_INFERENCE_FLAGS.get(str(backend).strip().lower())
    if allowed is None:
        return []
    unknown: list = []
    for key in (args or {}):
        name = str(key).lstrip("-").split("=", 1)[0].strip()
        if name and name not in allowed and name not in unknown:
            unknown.append(name)
    return unknown


def normalize_partitions(value: Any) -> str | None:
    """Normalise a ``slurm_partition`` config value to a comma-separated string.

    SLURM's ``sbatch -p`` natively accepts several partitions as a comma list
    (``-p gpu-el8,transform``) and schedules the job onto whichever one lets it
    start soonest. This lets a user list every GPU partition they may run on so
    inference jobs are not stuck behind one busy queue.

    Accepts any of:

    * a YAML list/tuple: ``[gpu-el8, transform]``
    * a comma- and/or whitespace-separated string: ``"gpu-el8, transform"``
    * a single partition string: ``"gpu-el8"`` (unchanged)
    * ``None`` / empty -> ``None`` (caller supplies its own fallback)

    Returns a de-duplicated, order-preserving comma-joined string (no spaces, so
    it survives ``shlex.quote`` unquoted and reaches ``sbatch`` verbatim), or
    ``None`` when no partition is given.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        # A single scalar; split on commas and any surrounding whitespace so both
        # "a,b", "a, b" and "a b" are accepted.
        items = str(value).replace(",", " ").split()
    names: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in names:
            names.append(name)
    return ",".join(names) if names else None


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


def bin_folds(
    fold_tokens: Iterable[tuple[str, int]],
    *,
    batch_size: int = 1,
    max_batch_tokens: int = 0,
) -> list[list[str]]:
    """Group folds into batches that share a single inference job (issue #48).

    Each batch is run by one ``run_structure_prediction.py`` invocation, which
    loads the model once and predicts the folds back to back. Batching trades
    finer-grained retries for far less queue wait and a single model-load per
    batch instead of one per fold.

    Folds are sorted by token count so a batch holds similarly sized folds: the
    batch's memory is sized from its largest fold and its walltime from the sum,
    so keeping sizes close stops a tiny fold from inheriting a huge fold's
    allocation (and clusters the many small folds the issue is about). The number
    of folds per batch is capped by ``batch_size``; the optional
    ``max_batch_tokens`` additionally caps the summed tokens per batch so total
    walltime stays within the partition limit. A single fold always forms a valid
    batch even if it alone exceeds ``max_batch_tokens``.

    ``batch_size <= 1`` returns one fold per batch in the original input order,
    i.e. the unbatched behaviour, so the default path is unchanged.
    """
    items = [(str(fold), int(tokens or 0)) for fold, tokens in fold_tokens]
    if batch_size <= 1:
        return [[fold] for fold, _ in items]

    ordered = sorted(items, key=lambda ft: (ft[1], ft[0]))
    cap = int(max_batch_tokens or 0)

    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for fold, tokens in ordered:
        too_many = len(current) >= batch_size
        too_big = cap > 0 and bool(current) and (current_tokens + tokens) > cap
        if current and (too_many or too_big):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(fold)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches
