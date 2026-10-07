"""Pin the copied inference flag tables.

These mirror ``alphapulldown/prediction/inference_flags.py``. The workflow parses on the
head node, where AlphaPulldown is not installed: it only exists inside the prediction
container, and CI does not install it either. So the recorded sets below make a change
to the copy deliberate and record what the original said when it was last checked, and
where AlphaPulldown is importable (a development environment, or ``PYTHONPATH`` pointing
at a checkout) the copy is also compared with the original itself.
"""

import importlib
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

_COMMON = SourceFileLoader(
    "common_flag_tables", str(Path(__file__).resolve().parents[1] / "workflow/rules/common.smk")
).load_module()

# Copied from alphapulldown/prediction/inference_flags.py in AlphaPulldown.
EXPECTED_COMMON = {
    "input", "output_directory", "data_directory", "features_directory",
    "protein_delimiter", "fold_backend", "random_seed", "storage_mode",
}
EXPECTED_AF2_LIKE = {
    "compress_result_pickles", "remove_result_pickles", "models_to_relax",
    "relax_best_score_threshold", "remove_keys_from_pickles",
    "convert_to_modelcif", "allow_resume",
    "num_cycle", "num_predictions_per_model", "pair_msa",
    "save_features_for_multimeric_object", "skip_templates",
    "msa_depth_scan", "multimeric_template", "model_names", "msa_depth",
    "description_file", "path_to_mmt", "threshold_clashes", "hb_allowance",
    "plddt_threshold", "desired_num_res", "desired_num_msa",
    "benchmark", "model_preset", "use_ap_style", "use_gpu_relax", "dropout",
    "jax_compilation_cache_dir",
}
EXPECTED_AF3 = {
    "jax_compilation_cache_dir", "buckets", "flash_attention_implementation",
    "num_diffusion_samples", "num_seeds", "debug_templates", "debug_msas",
    "num_recycles", "save_embeddings", "save_distogram", "use_ap_style",
    "convert_to_modelcif",
}


def test_tables_match_the_recorded_alphapulldown_sets():
    assert _COMMON._COMMON_INFERENCE_FLAGS == EXPECTED_COMMON
    assert _COMMON._AF2_LIKE_INFERENCE_FLAGS == EXPECTED_AF2_LIKE
    assert _COMMON._AF3_INFERENCE_FLAGS == EXPECTED_AF3
    assert _COMMON._ALPHALINK_EXTRA_FLAGS == {"crosslinks"}
    assert _COMMON._FAST_KERNEL_FLAGS == {"fast_kernels"}


def test_convert_to_modelcif_is_valid_on_both_backends():
    """The drift that made this test necessary."""
    for backend in ("alphafold2", "alphafold3"):
        args = {"--fold_backend": backend, "--convert_to_modelcif": True}
        assert _COMMON.unknown_inference_flags(args, backend) == []


def test_fast_kernels_is_valid_on_both_backends():
    for backend in ("alphafold2", "alphafold3"):
        args = {"--fold_backend": backend, "--fast_kernels": "auto"}
        assert _COMMON.unknown_inference_flags(args, backend) == []
    args = {"--fold_backend": "alphalink", "--fast_kernels": "auto"}
    assert _COMMON.unknown_inference_flags(args, "alphalink") == ["fast_kernels"]


def test_jax_compile_cache_is_valid_on_both_backends():
    for backend in ("alphafold2", "alphafold3"):
        args = {"--fold_backend": backend, "--jax_compilation_cache_dir": "/cache"}
        assert _COMMON.unknown_inference_flags(args, backend) == []


def _alphapulldown_inference_flags():
    """AlphaPulldown's flag module, wherever this release keeps it; skip if absent."""
    errors = []
    # alphapulldown.prediction since 2.9.1, the package root in 2.8.0.
    for name in ("alphapulldown.prediction.inference_flags", "alphapulldown.inference_flags"):
        try:
            return importlib.import_module(name)
        except ImportError as error:
            errors.append(f"{name}: {error}")
    pytest.skip("AlphaPulldown's inference_flags is not importable (" + "; ".join(errors) + ")")


def test_tables_match_alphapulldown_when_importable():
    """The per-backend allow sets equal AlphaPulldown's FLAGS_BY_BACKEND.

    Compared per backend rather than per named subset, so AlphaPulldown can regroup its
    sets (AF2_EXTRA_FLAGS became FAST_KERNEL_FLAGS) without this test noticing; only a
    change in what a backend accepts is drift.
    """
    module = _alphapulldown_inference_flags()
    theirs = {backend: set(flags) for backend, flags in module.FLAGS_BY_BACKEND.items()}
    ours = _COMMON.ALLOWED_INFERENCE_FLAGS
    package = importlib.import_module(module.__name__.split(".")[0])
    source = f"AlphaPulldown {getattr(package, '__version__', '?')} ({module.__file__})"

    assert set(ours) == set(theirs), f"backends differ from {source}"
    drift = {
        backend: {
            "only in common.smk": sorted(ours[backend] - theirs[backend]),
            "only in AlphaPulldown": sorted(theirs[backend] - ours[backend]),
        }
        for backend in sorted(ours)
        if ours[backend] != theirs[backend]
    }
    assert not drift, (
        f"common.smk's inference flag tables have drifted from {source}: {drift}. "
        "Update the sets in common.smk and the recorded sets in this file, or test "
        "against the AlphaPulldown this workflow release targets."
    )
