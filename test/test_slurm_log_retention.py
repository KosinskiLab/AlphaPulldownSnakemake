"""Exercise the executor's actual successful-job cleanup with our profile settings."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml


@pytest.mark.parametrize("use_profile", [False, True])
def test_successful_job_log_retention(tmp_path, monkeypatch, use_profile):
    slurm = pytest.importorskip("snakemake_executor_plugin_slurm")
    profile = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config/profiles/slurm/config.yaml").read_text()
    )
    # The old profile set the age cutoff only; the plugin still deleted each
    # successful job's log immediately. Compare that behavior with our profile.
    settings = slurm.ExecutorSettings(
        delete_logfiles_older_than=profile["slurm-delete-logfiles-older-than"],
        keep_successful_logs=profile.get("slurm-keep-successful-logs", False) if use_profile else False,
    )
    logfile = tmp_path / "123.log"
    logfile.write_text("inference completed\n")
    job = SimpleNamespace(external_jobid="123", aux={"slurm_logfile": logfile})
    executor = SimpleNamespace(
        workflow=SimpleNamespace(executor_settings=settings),
        logger=logging.getLogger(__name__), run_uuid="log-retention-test",
        get_status_command=lambda: "sacct", status_rate_limiter=AsyncMock(),
        job_stati=AsyncMock(return_value=({"123": "COMPLETED"}, 0.01)),
        report_job_success=Mock(),
    )
    monkeypatch.setattr(slurm, "get_min_job_age", lambda: 300)
    monkeypatch.setattr(slurm, "is_query_tool_available", lambda name: True)
    monkeypatch.setattr(slurm, "query_job_status_sacct", lambda run_uuid: "mock status")

    async def complete():
        return [job async for job in slurm.Executor.check_active_jobs(executor, [job])]

    assert asyncio.run(complete()) == []
    executor.report_job_success.assert_called_once_with(job)
    assert logfile.exists() == use_profile
    if use_profile:
        # Shutdown cleanup must also retain it, even if the file is old.
        executor.slurm_logdir = tmp_path
        slurm.Executor.clean_old_logs(executor)
        assert logfile.read_text() == "inference completed\n"
