"""Unit tests for ``batch_inference_args`` (issue #48 batching, backend flag gating).

Regression guard: ``run_structure_prediction.py`` hard-errors on a flag its backend does
not accept (``ValueError: not supported by backend '<name>'``). Adding the wrong flag once
broke ALL batched AlphaFold2 inference. ``--allow_resume`` is AlphaFold2-only;
``--jax_compilation_cache_dir`` is accepted by both JAX backends (AlphaFold2 since
AlphaPulldown 2.8.0). These tests pin each flag to the backends that accept it.

``common.smk`` is plain Python, loaded by path. Run with
``python test/test_batch_inference_args.py`` or ``pytest test/test_batch_inference_args.py``.
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
batch_inference_args = common.batch_inference_args

_CACHE = "/out/.jax_compilation_cache"


def test_af2_batch_gets_allow_resume_and_jax_cache():
    args = batch_inference_args(
        {"--fold_backend": "alphafold2"},
        backend="alphafold2",
        batch_size=2,
        jax_cache_dir=_CACHE,
    )
    assert args["--allow_resume"] == "true"
    assert args["--jax_compilation_cache_dir"] == _CACHE


def test_af3_batch_gets_jax_cache_not_allow_resume():
    args = batch_inference_args(
        {"--fold_backend": "alphafold3"},
        backend="alphafold3",
        batch_size=2,
        jax_cache_dir=_CACHE,
    )
    assert args["--jax_compilation_cache_dir"] == _CACHE
    # the AF2-only flag must NOT be present for AF3 (it would ValueError at runtime)
    assert "--allow_resume" not in args


def test_batch_size_one_gets_the_cache_only():
    # A fresh process per fold is where the shared cache saves the most compilation.
    for backend in ("alphafold2", "alphafold3"):
        base = {"--fold_backend": backend}
        args = batch_inference_args(
            base, backend=backend, batch_size=1, jax_cache_dir=_CACHE
        )
        assert args == {**base, "--jax_compilation_cache_dir": _CACHE}


def test_no_cache_directory_adds_no_cache_flag():
    # The Snakefile passes "" for AlphaLink, a PyTorch backend.
    args = batch_inference_args(
        {"--fold_backend": "alphalink"}, backend="alphafold2", batch_size=2, jax_cache_dir=""
    )
    assert "--jax_compilation_cache_dir" not in args
    assert args["--allow_resume"] == "true"


def test_user_can_turn_the_cache_off():
    for off in (False, None, "", "false", "None", "null"):
        args = batch_inference_args(
            {"--fold_backend": "alphafold3", "--jax_compilation_cache_dir": off},
            backend="alphafold3",
            batch_size=4,
            jax_cache_dir=_CACHE,
        )
        assert "--jax_compilation_cache_dir" not in args, off


def test_user_values_are_preserved():
    # explicit user settings win over the workflow defaults (setdefault semantics)
    args = batch_inference_args(
        {"--fold_backend": "alphafold3", "--jax_compilation_cache_dir": "/custom"},
        backend="alphafold3",
        batch_size=4,
        jax_cache_dir=_CACHE,
    )
    assert args["--jax_compilation_cache_dir"] == "/custom"


def test_does_not_mutate_input():
    base = {"--fold_backend": "alphafold2", "--jax_compilation_cache_dir": "false"}
    batch_inference_args(base, backend="alphafold2", batch_size=2, jax_cache_dir=_CACHE)
    assert base == {"--fold_backend": "alphafold2", "--jax_compilation_cache_dir": "false"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all batch_inference_args tests passed")
