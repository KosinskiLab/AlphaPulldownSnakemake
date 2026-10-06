"""The default prediction images accept the inference flags the README documents.

``run_structure_prediction.py`` aborts on a flag its release does not know, so a README
that documents a flag the default image rejects sends users straight into a failed job.
``MINIMUM_ALPHAPULLDOWN`` records, per backend, the oldest AlphaPulldown release that
accepts each flag newer than the rest; the test checks every default image tag in
``config/config.yaml`` against the flags the README documents or the workflow adds.

A flag no AlphaPulldown release accepts yet is written ``">LAST"``: any release after
LAST, the newest one that rejects it. Its backend's case is then an expected failure.
The release that ships the flag bumps the tags in ``config/config.yaml`` and writes its
version here as ``">=X.Y.Z"``. Doing only one of the two fails the suite (the strict
xfail passes, or the tags are too old), so neither can be forgotten.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

_REPOSITORY = Path(__file__).resolve().parents[1]
_README = (_REPOSITORY / "README.md").read_text(encoding="utf-8")
_CONFIG_TEXT = (_REPOSITORY / "config" / "config.yaml").read_text(encoding="utf-8")

_loader = importlib.machinery.SourceFileLoader(
    "aps_common_tags", str(_REPOSITORY / "workflow" / "rules" / "common.smk")
)
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
common = importlib.util.module_from_spec(_spec)
_loader.exec_module(common)

# Oldest AlphaPulldown release accepting each flag, per backend. Flags not listed are
# accepted by every release this workflow supports.
MINIMUM_ALPHAPULLDOWN = {
    "alphafold2": {
        "jax_compilation_cache_dir": ">=2.8.0",
        # KosinskiLab/AlphaPulldown#645, merged after 2.9.1, not yet released.
        "fast_kernels": ">2.9.1",
    },
    "alphafold3": {
        # AlphaPulldown exp/af3-v3.0.4 (AF3 fused triangle kernels), not yet released.
        "fast_kernels": ">2.9.1",
    },
}

_BACKEND_SECTIONS = {"alphafold2": "AlphaFold2 flags", "alphafold3": "AlphaFold3 flags"}
_IMAGE_TAG = re.compile(r"\balphafold([23]):(\d+(?:\.\d+)+)\b")


def _version(text):
    return tuple(int(part) for part in text.split("."))


def _satisfies(version, requirement):
    if requirement.startswith(">="):
        return _version(version) >= _version(requirement[2:])
    if requirement.startswith(">"):
        return _version(version) > _version(requirement[1:])
    raise ValueError(f"requirement must start with >= or >: {requirement!r}")


def _unreleased(backend):
    return sorted(
        flag for flag, requirement in MINIMUM_ALPHAPULLDOWN[backend].items()
        if not requirement.startswith(">=")
    )


def _readme_flags(backend):
    """Flag names in the README's collapsed '<backend> flags' YAML example."""
    section = re.search(
        rf"<summary>{re.escape(_BACKEND_SECTIONS[backend])}</summary>\s*```yaml\n(.*?)```",
        _README,
        re.DOTALL,
    )
    assert section, f"README has no '{_BACKEND_SECTIONS[backend]}' YAML block"
    return set(re.findall(r"^\s*--(\w+):", section.group(1), re.MULTILINE))


def _workflow_added_flags(backend):
    """Flags the workflow adds on its own, in its default configuration."""
    added = common.batch_inference_args(
        {}, backend=backend, batch_size=2, jax_cache_dir="/cache"
    )
    return {flag.lstrip("-") for flag in added}


def _default_images():
    """(backend, tag, line) for every alphafold image tag config/config.yaml names: the
    prediction_container default and the image its comments say to switch to."""
    return [
        (f"alphafold{match.group(1)}", match.group(2), line.strip())
        for line in _CONFIG_TEXT.splitlines()
        for match in _IMAGE_TAG.finditer(line)
    ]


def _image_cases():
    cases = []
    for backend, tag, line in _default_images():
        marks = []
        if _unreleased(backend):
            marks.append(pytest.mark.xfail(
                strict=True,
                reason=(
                    f"AlphaPulldown has not released {', '.join(_unreleased(backend))} "
                    f"for {backend} yet. The release that does bumps the image tags in "
                    "config/config.yaml and records its version in MINIMUM_ALPHAPULLDOWN."
                ),
            ))
        cases.append(pytest.param(backend, tag, line, marks=marks, id=f"{backend}:{tag}"))
    return cases


def test_config_names_a_pinned_default_image_for_each_backend():
    """Guards the parametrization below against matching nothing."""
    default = yaml.safe_load(_CONFIG_TEXT)["prediction_container"]
    assert _IMAGE_TAG.search(default), f"prediction_container is not a version tag: {default}"
    assert {backend for backend, _tag, _line in _default_images()} == set(_BACKEND_SECTIONS)


@pytest.mark.parametrize("backend", sorted(_BACKEND_SECTIONS))
def test_readme_documents_only_flags_the_backend_accepts(backend):
    documented = _readme_flags(backend)
    assert documented
    assert documented <= common.ALLOWED_INFERENCE_FLAGS[backend], sorted(
        documented - common.ALLOWED_INFERENCE_FLAGS[backend]
    )


@pytest.mark.parametrize("backend", sorted(_BACKEND_SECTIONS))
def test_minimum_versions_cover_documented_flags(backend):
    """Every flag with a minimum version is one the README documents or the workflow adds,
    so a renamed or dropped flag cannot leave a requirement nothing checks."""
    relevant = _readme_flags(backend) | _workflow_added_flags(backend)
    assert set(MINIMUM_ALPHAPULLDOWN[backend]) <= relevant


@pytest.mark.parametrize("backend,tag,line", _image_cases())
def test_default_image_accepts_the_documented_flags(backend, tag, line):
    relevant = _readme_flags(backend) | _workflow_added_flags(backend)
    too_old = {
        flag: requirement
        for flag, requirement in sorted(MINIMUM_ALPHAPULLDOWN[backend].items())
        if flag in relevant and not _satisfies(tag, requirement)
    }
    assert not too_old, (
        f"config/config.yaml names {backend}:{tag} ({line!r}), but the README documents "
        f"flags it rejects; they need AlphaPulldown {too_old}"
    )
