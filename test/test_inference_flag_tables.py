"""Pin the copied inference flag tables.

These mirror ``alphapulldown/inference_flags.py``, which cannot be imported here: the
workflow parses on the head node and AlphaPulldown only exists inside the prediction
container. A test that cannot reach the original can at least make a change to the copy
deliberate, and record what the original said when it was last checked.
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path

_COMMON = SourceFileLoader(
    "common_flag_tables", str(Path(__file__).resolve().parents[1] / "workflow/rules/common.smk")
).load_module()

# Copied from alphapulldown/inference_flags.py, AlphaPulldown feat/resident-inference.
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


def test_convert_to_modelcif_is_valid_on_both_backends():
    """The drift that made this test necessary."""
    for backend in ("alphafold2", "alphafold3"):
        args = {"--fold_backend": backend, "--convert_to_modelcif": True}
        assert _COMMON.unknown_inference_flags(args, backend) == []


def test_jax_compile_cache_is_valid_on_both_backends():
    for backend in ("alphafold2", "alphafold3"):
        args = {"--fold_backend": backend, "--jax_compilation_cache_dir": "/cache"}
        assert _COMMON.unknown_inference_flags(args, backend) == []
