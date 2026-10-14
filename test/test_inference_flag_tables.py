"""The inference flag tables in common.smk, generated from AlphaPulldown.

The workflow parses on the head node, where AlphaPulldown is not installed, so
``ALLOWED_INFERENCE_FLAGS`` is a generated copy of AlphaPulldown's ``FLAGS_BY_BACKEND``
that records the commit it came from (tools/generate_inference_flag_tables.py). These
tests pin the generator's contract on a throwaway checkout; CI's
``inference-flag-tables`` job regenerates the block from the recorded AlphaPulldown
commit itself and fails on any difference.
"""

import os
import re
import shutil
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
COMMON_SMK = REPOSITORY / "workflow" / "rules" / "common.smk"
GENERATOR = REPOSITORY / "tools" / "generate_inference_flag_tables.py"

_COMMON = SourceFileLoader("common_flag_tables", str(COMMON_SMK)).load_module()

# The AlphaPulldown the tables come from: main after 2.10.0, where AF3 accepts --allow_resume.
SHIPPED_ALPHAPULLDOWN_COMMIT = "228c5eed6c7066467e46b87fb3054cc2c2e73ae1"

# Shaped like AlphaPulldown's inference_flags.py: named sets, unions, an annotated dict.
_SOURCE = '''
from typing import Mapping

COMMON_FLAGS = frozenset({"input", "fold_backend"})
AF2_LIKE_FLAGS = frozenset({"num_cycle", "allow_resume"})
AF3_FLAGS = frozenset({"buckets", "num_seeds"})
FAST_KERNEL_FLAGS = frozenset({"fast_kernels"})

FLAGS_BY_BACKEND: Mapping[str, frozenset] = {
    "alphafold2": COMMON_FLAGS | AF2_LIKE_FLAGS | FAST_KERNEL_FLAGS,
    "alphafold3": COMMON_FLAGS | AF3_FLAGS | FAST_KERNEL_FLAGS,
}
'''


def _git(checkout, *args):
    return subprocess.run(
        ["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(checkout, message):
    _git(checkout, "add", "-A")
    _git(
        checkout, "-c", "user.name=test", "-c", "user.email=test@example.org",
        "commit", "-q", "-m", message,
    )


def _checkout(tmp_path, source=_SOURCE):
    """A committed AlphaPulldown-shaped checkout; returns it and its commit."""
    checkout = tmp_path / "alphapulldown-checkout"
    module = checkout / "alphapulldown" / "prediction" / "inference_flags.py"
    module.parent.mkdir(parents=True)
    module.write_text(source, encoding="utf-8")
    _git(checkout.parent, "init", "-q", str(checkout))
    _commit(checkout, "flags")
    return checkout, _git(checkout, "rev-parse", "HEAD")


def _generate(checkout, target, *flags):
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--source-checkout", str(checkout),
         "--target", str(target), *flags],
        capture_output=True, text=True,
    )


@pytest.fixture
def target(tmp_path):
    """A copy of common.smk to generate into."""
    copy = tmp_path / "common.smk"
    shutil.copyfile(COMMON_SMK, copy)
    return copy


def _tables(path):
    module = SourceFileLoader(f"tables_{abs(hash(str(path)))}", str(path)).load_module()
    return module.ALLOWED_INFERENCE_FLAGS


def test_common_smk_records_the_alphapulldown_commit_its_tables_came_from():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--print-recorded-commit"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == SHIPPED_ALPHAPULLDOWN_COMMIT
    assert set(_COMMON.ALLOWED_INFERENCE_FLAGS) == {"alphafold2", "alphafold3", "alphalink"}


def test_generation_is_deterministic_and_check_writes_nothing(tmp_path, target):
    checkout, commit = _checkout(tmp_path)

    assert _generate(checkout, target).returncode == 0
    generated = target.read_text()
    assert _generate(checkout, target).returncode == 0
    assert target.read_text() == generated
    mtime = os.stat(target).st_mtime_ns

    check = _generate(checkout, target, "--check")
    assert check.returncode == 0, check.stderr
    assert os.stat(target).st_mtime_ns == mtime and target.read_text() == generated

    assert f"KosinskiLab/AlphaPulldown@{commit}" in generated
    assert _tables(target) == {
        "alphafold2": {"input", "fold_backend", "num_cycle", "allow_resume", "fast_kernels"},
        "alphafold3": {"input", "fold_backend", "buckets", "num_seeds", "fast_kernels"},
    }
    # Sorted, so a regenerated block differs only where a flag did.
    block = generated[generated.index("ALLOWED_INFERENCE_FLAGS = {"):]
    alphafold2 = re.findall(r'^        "(\w+)",$', block.split('"alphafold3"')[0], re.MULTILINE)
    assert alphafold2 == sorted(alphafold2)


@pytest.mark.parametrize(
    "edit",
    [
        ('frozenset({"num_cycle", "allow_resume"})', 'frozenset({"num_cycle"})'),
        ('"buckets", "num_seeds"', '"buckets", "num_seed"'),
    ],
    ids=["removed", "renamed"],
)
def test_check_fails_when_alphapulldown_changes_a_flag(tmp_path, target, edit):
    checkout, _ = _checkout(tmp_path)
    assert _generate(checkout, target).returncode == 0

    module = checkout / "alphapulldown" / "prediction" / "inference_flags.py"
    module.write_text(module.read_text().replace(*edit))
    _commit(checkout, "change a flag")
    before = target.read_text()

    check = _generate(checkout, target, "--check")

    assert check.returncode == 1
    assert target.read_text() == before
    assert "differs from KosinskiLab/AlphaPulldown@" in check.stderr


def test_an_uncommitted_source_is_refused(tmp_path, target):
    checkout, _ = _checkout(tmp_path)
    module = checkout / "alphapulldown" / "prediction" / "inference_flags.py"
    module.write_text(module.read_text().replace("num_seeds", "num_samples"))

    result = _generate(checkout, target, "--check")

    assert result.returncode == 2
    assert "uncommitted changes" in result.stderr


@pytest.mark.parametrize("problem", ["no checkout", "no FLAGS_BY_BACKEND"])
def test_a_missing_source_fails_rather_than_skips(tmp_path, target, problem):
    if problem == "no checkout":
        checkout = tmp_path / "missing"
    else:
        checkout, _ = _checkout(tmp_path, source="COMMON_FLAGS = frozenset({'input'})\n")

    result = _generate(checkout, target, "--check")

    assert result.returncode == 2
    assert result.stderr.startswith("error:")


def test_the_tables_need_no_alphapulldown_to_parse():
    """The head node parses common.smk without AlphaPulldown installed."""
    probe = (
        "import sys\n"
        "from importlib.machinery import SourceFileLoader\n"
        f"module = SourceFileLoader('common', {str(COMMON_SMK)!r}).load_module()\n"
        "assert 'alphapulldown' not in sys.modules, 'common.smk imported AlphaPulldown'\n"
        "assert module.unknown_inference_flags({'--buckets': 1}, 'alphafold3') == []\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_convert_to_modelcif_is_valid_on_both_backends():
    """The drift that made the tables a tested copy."""
    for backend in ("alphafold2", "alphafold3"):
        args = {"--fold_backend": backend, "--convert_to_modelcif": True}
        assert _COMMON.unknown_inference_flags(args, backend) == []


def test_fast_kernels_is_valid_on_both_backends():
    for backend in ("alphafold2", "alphafold3"):
        args = {"--fold_backend": backend, "--fast_kernels": "auto"}
        assert _COMMON.unknown_inference_flags(args, backend) == []
    args = {"--fold_backend": "alphalink", "--fast_kernels": "auto"}
    assert _COMMON.unknown_inference_flags(args, "alphalink") == ["fast_kernels"]


def test_allow_resume_is_valid_on_both_backends():
    for backend in ("alphafold2", "alphafold3"):
        args = {"--fold_backend": backend, "--allow_resume": False}
        assert _COMMON.unknown_inference_flags(args, backend) == []


def test_jax_compile_cache_is_valid_on_both_backends():
    for backend in ("alphafold2", "alphafold3"):
        args = {"--fold_backend": backend, "--jax_compilation_cache_dir": "/cache"}
        assert _COMMON.unknown_inference_flags(args, backend) == []
