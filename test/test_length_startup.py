"""Startup progress and restart safety for sequence-length resolution."""

import importlib.machinery
import importlib.util
import logging
import io
import urllib.error
from pathlib import Path

import pytest


_COMMON = Path(__file__).resolve().parents[1] / "workflow/rules/common.smk"
_loader = importlib.machinery.SourceFileLoader("length_startup_common", str(_COMMON))
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
common = importlib.util.module_from_spec(_spec)
_loader.exec_module(common)
logger = logging.getLogger(__name__)


def test_progress_is_visible_during_lookups_and_cache_survives_restart(
    tmp_path, monkeypatch, caplog,
):
    now = [0.0]
    monkeypatch.setattr(common.time, "monotonic", lambda: now[0])
    caplog.set_level(logging.INFO, logger=logger.name)
    path = tmp_path / ".sequence_lengths.tsv"
    cache = {}
    requested = []

    def resolve(name):
        assert "Resolving lengths for 3 unique proteins" in caplog.text
        if name == "B":
            # This is checked DURING resolution, not just after it returns.
            assert "Checked 1/3 proteins; elapsed 31s; current protein: B" in caplog.text
            assert path.read_text() == "A\t123\n"
            raise KeyboardInterrupt
        requested.append(name)
        now[0] += 31
        cache[name] = 123
        return 123

    with pytest.raises(KeyboardInterrupt):
        common.resolve_sequence_lengths(
            ["A", "A", "B", "C"], resolve,
            lambda: common.persist_sequence_lengths(path, cache), logger,
        )
    assert "Length resolution complete" not in caplog.text
    assert requested == ["A"]

    # A fresh invocation starts from the saved file and never fetches A again.
    cache = {name: int(length) for name, length in
             (line.split("\t") for line in path.read_text().splitlines())}

    def resumed_resolve(name):
        if name not in cache:
            requested.append(name)
            cache[name] = 456
        return cache[name]

    result = common.resolve_sequence_lengths(
        ["A", "B", "A", "C"], resumed_resolve,
        lambda: common.persist_sequence_lengths(path, cache), logger,
    )
    assert result == {"A": 123, "B": 456, "C": 456}
    assert requested == ["A", "B", "C"]
    assert "3/3 proteins checked, 3 resolved, 0 unknown" in caplog.text
    assert path.read_text() == "A\t123\nB\t456\nC\t456\n"


def test_interruption_before_first_checkpoint_still_saves_successes(tmp_path):
    path = tmp_path / ".sequence_lengths.tsv"
    cache = {}

    def resolve(name):
        if name == "B":
            raise KeyboardInterrupt
        cache[name] = 10
        return 10

    with pytest.raises(KeyboardInterrupt):
        common.resolve_sequence_lengths(
            ["A", "B"], resolve,
            lambda: common.persist_sequence_lengths(path, cache), logger,
        )
    assert path.read_text() == "A\t10\n"


def test_failed_atomic_write_preserves_previous_cache(tmp_path, monkeypatch):
    path = tmp_path / ".sequence_lengths.tsv"
    path.write_text("A\t10\n")

    def fail_replace(source, target):
        assert Path(source).read_text() == "A\t10\nB\t20\n"
        assert Path(target).read_text() == "A\t10\n"
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(common.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        common.persist_sequence_lengths(path, {"A": 10, "B": 20})
    assert path.read_text() == "A\t10\n"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("names", [[], ["A", "A", "B"]])
def test_fast_offline_scan_reports_unknowns_without_progress_spam(names, caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    checkpoints = []
    results = common.resolve_sequence_lengths(
        names, lambda name: 12 if name == "A" else None,
        lambda: checkpoints.append(True), logger, fetch_uniprot=False,
    )
    assert results == ({"A": 12, "B": None} if names else {})
    assert "UniProt lookups disabled" in caplog.text
    assert ("1 resolved, 1 unknown" if names else "0 resolved, 0 unknown") in caplog.text
    assert len(caplog.records) == 2  # Start + completion even for an empty scan.
    assert checkpoints == [True]


def test_uniprot_failure_categories_are_summarized_and_cached(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    diagnostics = common.UniprotLengthDiagnostics(logger)
    calls = []
    failures = {
        "timeout": TimeoutError(),
        "wrapped_timeout": urllib.error.URLError(TimeoutError()),
        "missing": urllib.error.HTTPError("", 404, "missing", {}, None),
        "gone": urllib.error.HTTPError("", 410, "gone", {}, None),
        "server": urllib.error.HTTPError("", 503, "unavailable", {}, None),
        "network": urllib.error.URLError("unreachable"),
    }

    def urlopen(url, timeout):
        name = url.rsplit("/", 1)[-1].removesuffix(".fasta")
        calls.append(name)
        if name in failures:
            raise failures[name]
        return io.BytesIO(b">empty\n" if name == "empty" else b">ok\nACDE\n")

    common.fetch_uniprot_length.cache_clear()
    monkeypatch.setattr(common.urllib.request, "urlopen", urlopen)

    def resolve(name):
        return common.fetch_uniprot_length(name, on_failure=diagnostics.record) or None

    results = common.resolve_sequence_lengths(
        [*failures, "empty", "ok", "missing"], resolve, lambda: None, logger,
        report_failures=diagnostics.report,
    )
    assert results["ok"] == 4
    assert all(results[name] is None for name in failures)
    assert diagnostics.counts == {"timeout": 2, "not found": 2, "other errors": 3}
    assert "timeout=2" in caplog.text
    assert "not found=2" in caplog.text
    assert "server (HTTP 503)" in caplog.text
    assert "empty (empty sequence response)" in caplog.text
    assert "8/8 proteins checked, 1 resolved, 7 unknown" in caplog.text
    assert resolve("missing") is None
    diagnostics.report()
    assert calls.count("missing") == 1
    assert sum("[uniprot]" in record.message for record in caplog.records) == 1


def test_failure_summary_is_bounded_and_reported_before_scan_finishes(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    now = [0.0]
    monkeypatch.setattr(common.time, "monotonic", lambda: now[0])
    diagnostics = common.UniprotLengthDiagnostics(logger)

    def resolve(name):
        if name == "last":
            assert "timeout=20" in caplog.text
            assert "Length resolution complete" not in caplog.text
            return 4
        diagnostics.record(name, "timeout", "request timed out")
        now[0] += 2

    common.resolve_sequence_lengths(
        [*(str(n) for n in range(20)), "last"], resolve, lambda: None, logger,
        interval=40, report_failures=diagnostics.report,
    )
    assert len(diagnostics.examples["timeout"]) == 3
    assert sum("[uniprot]" in record.message for record in caplog.records) == 1


def test_feature_discovery_reports_unique_reuse_missing_directory_and_progress(
    tmp_path, caplog, monkeypatch,
):
    caplog.set_level(logging.INFO, logger=logger.name)
    good = tmp_path / "features"
    good.mkdir()
    (good / "A.pkl").write_text("fixture")
    (good / "unrelated.pkl").write_text("fixture")
    absent = tmp_path / "missing"
    ticks = iter(range(0, 1000, 31))
    monkeypatch.setattr(common.time, "monotonic", lambda: next(ticks))
    found = common.discover_precomputed_features(
        {"A.pkl", "B.pkl"}, [good, absent, good], logger,
    )
    assert found == [str(good / "A.pkl"), str(good / "A.pkl")]
    assert f"Scanning directory: {good}" in caplog.text
    assert f"does not exist: {absent}" in caplog.text
    assert "checked 1/2 entries" in caplog.text
    assert "Discovery complete: 1/2" in caplog.text
    assert "1 not found there" in caplog.text


def test_unreadable_feature_directory_logs_path_and_preserves_error(tmp_path, caplog, monkeypatch):
    def denied(path):
        raise PermissionError("simulated denial")

    monkeypatch.setattr(common.os, "listdir", denied)
    with pytest.raises(PermissionError, match="simulated denial"):
        common.discover_precomputed_features({"A.pkl"}, [tmp_path], logger)
    assert f"Cannot scan directory {tmp_path}" in caplog.text
