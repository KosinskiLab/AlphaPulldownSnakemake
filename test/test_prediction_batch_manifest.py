"""Behavioral tests for the resident-inference manifest seam."""

import importlib.machinery
import importlib.util
import json
import re
from pathlib import Path


_COMMON = Path(__file__).resolve().parents[1] / "workflow" / "rules" / "common.smk"
_spec = importlib.util.spec_from_loader(
    "aps_common_prediction_batch",
    importlib.machinery.SourceFileLoader("aps_common_prediction_batch", str(_COMMON)),
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


def test_batch_identity_changes_with_members_but_preserves_single_fold_identity():
    assert common.prediction_batch_id(["A+B"]) == "A+B"

    first = common.prediction_batch_id(["A+B", "C+D"])
    changed = common.prediction_batch_id(["A+B", "E+F"])

    assert first.startswith("batch-")
    assert changed.startswith("batch-")
    assert first != changed
    assert first == common.prediction_batch_id(["A+B", "C+D"])


def test_resident_batch_identity_is_a_bounded_filesystem_component():
    batch_id = common.prediction_batch_id(["A" * 250, "B" * 250])

    assert re.fullmatch(r"batch-[0-9a-f]{64}", batch_id)
    assert len(batch_id.encode("utf-8")) < 255


def test_manifest_preserves_batch_order_and_uses_manifest_relative_outputs(tmp_path):
    manifest = tmp_path / "output" / ".prediction_batches" / "small.jsonl"
    prediction_root = tmp_path / "output" / "predictions"

    contents = common.prediction_batch_manifest(
        [
            ("A+B", "A+B", prediction_root / "A+B"),
            ("C+D", "C+D", prediction_root / "C+D"),
        ],
        manifest,
    )

    assert [json.loads(line) for line in contents.splitlines()] == [
        {
            "job_id": "A+B",
            "input": "A+B",
            "output_directory": "../predictions/A+B",
        },
        {
            "job_id": "C+D",
            "input": "C+D",
            "output_directory": "../predictions/C+D",
        },
    ]
