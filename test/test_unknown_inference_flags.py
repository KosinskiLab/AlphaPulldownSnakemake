"""Unit tests for ``unknown_inference_flags`` (parse-time backend-flag guard).

Warns early when ``structure_inference_arguments`` contains a flag the selected backend
does not accept, instead of failing deep in a Slurm job. Mirrors the container's
``_validate_flags_for_backend`` allow-sets.

Run with ``python test/test_unknown_inference_flags.py`` or pytest.
"""

import importlib.machinery
import importlib.util
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[1] / "workflow" / "rules" / "common.smk"
_spec = importlib.util.spec_from_loader(
    "aps_common", importlib.machinery.SourceFileLoader("aps_common", str(_COMMON))
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
unknown_inference_flags = common.unknown_inference_flags


def test_af2_flags_all_accepted():
    args = {"--fold_backend": "alphafold2", "--num_cycle": 3, "--allow_resume": "true"}
    assert unknown_inference_flags(args, "alphafold2") == []


def test_jax_cache_dir_accepted_on_af2():
    # AF2 inference is JAX-compiled, so newer containers accept the compile cache flag.
    args = {"--fold_backend": "alphafold2", "--jax_compilation_cache_dir": "/c"}
    assert unknown_inference_flags(args, "alphafold2") == []


def test_af2_only_flag_flagged_on_af3():
    args = {"--fold_backend": "alphafold3", "--allow_resume": "true"}
    assert unknown_inference_flags(args, "alphafold3") == ["allow_resume"]


def test_af3_flags_all_accepted():
    args = {"--fold_backend": "alphafold3", "--num_diffusion_samples": 5,
            "--jax_compilation_cache_dir": "/c", "--use_ap_style": False}
    assert unknown_inference_flags(args, "alphafold3") == []


def test_alphalink_accepts_crosslinks_and_af2_flags():
    args = {"--fold_backend": "alphalink", "--crosslinks": "x.pkl", "--num_cycle": 3}
    assert unknown_inference_flags(args, "alphalink") == []
    # but crosslinks is NOT an AlphaFold2 flag
    assert unknown_inference_flags({"--crosslinks": "x.pkl"}, "alphafold2") == ["crosslinks"]


def test_leading_dashes_and_equals_are_stripped():
    # keys may arrive with or without dashes, or with an inline value
    assert unknown_inference_flags({"allow_resume": "true"}, "alphafold3") == ["allow_resume"]
    assert unknown_inference_flags({"--bogus=1": "x"}, "alphafold3") == ["bogus"]


def test_unknown_backend_stays_silent():
    # can't judge an unrecognised backend -> no false warnings
    assert unknown_inference_flags({"--anything": 1}, "some-future-backend") == []


def test_backend_name_case_insensitive():
    assert unknown_inference_flags({"--allow_resume": "t"}, "AlphaFold3") == ["allow_resume"]


def test_order_preserved_and_deduped():
    args = {"--allow_resume": "t", "--models_to_relax": "best", "--allow_resume2": "t"}
    # only genuinely-unknown AF3 flags, in order
    out = unknown_inference_flags(args, "alphafold3")
    assert out == ["allow_resume", "models_to_relax", "allow_resume2"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all unknown_inference_flags tests passed")
