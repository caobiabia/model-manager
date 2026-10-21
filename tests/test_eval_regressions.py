"""Offline regressions for the H200 evaluation changes; no model calls."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
from eval import common
from eval.runner import summary_status


class SummaryStatusTests(unittest.TestCase):
    config = {"model_keys": ["model"], "bench_ids": ["bench"], "limit": 0}

    def status(self, **fields):
        result = {"model_key": "model", "bench_id": "bench", **fields}
        return summary_status({"results": [result]}, self.config)

    def test_missing_model_bench_pair_is_interrupted(self):
        self.assertEqual(summary_status({"results": []}, self.config), "interrupted")

    def test_persisted_complete_pair_is_completed(self):
        self.assertEqual(self.status(unattempted=0, failed_pending=0), "completed")

    def test_infrastructure_failures_do_not_count_as_completion(self):
        for field in ("unreachable", "aborted", "failed_pending", "unattempted"):
            with self.subTest(field=field):
                self.assertEqual(self.status(**{field: 1}), "interrupted")

    def test_intentionally_limited_run_may_complete(self):
        config = {**self.config, "limit": 10}
        summary = {"results": [{"model_key": "model", "bench_id": "bench",
                                "unattempted": 90}]}
        self.assertEqual(summary_status(summary, config), "completed")
        summary["results"][0]["failed_pending"] = 1
        self.assertEqual(summary_status(summary, config), "interrupted")

    def test_explicit_error_or_stop_survives_reload(self):
        for status in ("error", "stopped", "aborted", "running"):
            with self.subTest(status=status):
                self.assertEqual(summary_status({"status": status}, self.config), status)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        common.clear_abort()
        self.client = SimpleNamespace(base_url="http://offline.test/v1")
        self.patches = [
            patch.object(common, "_server_max_tokens", return_value=None),
            patch.object(common.time, "monotonic", side_effect=lambda: self.now),
            patch.object(common.time, "sleep", side_effect=self.sleep),
        ]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        self.addCleanup(common.clear_abort)

    def sleep(self, seconds):
        self.now += seconds

    def call(self, **kwargs):
        return common.call_model(self.client, "model", [], "zero_shot", False, **kwargs)

    def test_connection_recovers_without_charging_healthy_generation_time(self):
        calls = 0

        def complete(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.now += 600
                raise common.ConnectionFailure("tunnel down")
            return "A", "", None

        with patch.object(common, "_stream_completion", side_effect=complete):
            self.assertEqual(self.call(connection_recovery_timeout=2)[0], "A")
        self.assertEqual(calls, 2)

    def test_outage_budget_expires(self):
        with patch.object(common, "_stream_completion",
                          side_effect=common.ConnectionFailure("offline")) as stream:
            with self.assertRaises(common.ConnectionFailure):
                self.call(connection_recovery_timeout=0.5)
        self.assertEqual(stream.call_count, 2)
        self.assertAlmostEqual(self.now, 0.5)

    def test_generation_timeout_is_not_retried(self):
        with patch.object(common, "_stream_completion",
                          side_effect=common.GenerationTimeout("slow")) as stream:
            with self.assertRaises(common.GenerationTimeout):
                self.call(connection_recovery_timeout=180)
        self.assertEqual(stream.call_count, 1)

    def test_user_stop_interrupts_recovery_wait(self):
        def stop_during_wait(seconds):
            self.now += seconds
            common.request_abort()

        with patch.object(common.time, "sleep", side_effect=stop_during_wait), \
             patch.object(common, "_stream_completion",
                          side_effect=common.ConnectionFailure("offline")) as stream:
            with self.assertRaises(common.RequestAborted):
                self.call(connection_recovery_timeout=180)
        self.assertEqual(stream.call_count, 1)


class VllmEnvironmentTests(unittest.TestCase):
    def test_finds_project_and_legacy_shared_environments(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            project = workspace / "model-console"
            shared = workspace / ".venv/bin/vllm"
            local = project / ".venv/bin/vllm"
            for executable in (shared, local):
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.touch()
                executable.chmod(0o755)
            with patch.object(app, "APP_DIR", project), \
                 patch.object(app, "_workspace", workspace), \
                 patch.object(app.sys, "executable", str(workspace / "python/bin/python")):
                self.assertEqual(app._vllm_executable(), str(local))
                local.unlink()
                self.assertEqual(app._vllm_executable(), str(shared))

    def test_active_environment_takes_priority(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "vllm"
            executable.touch()
            executable.chmod(0o755)
            with patch.object(app.sys, "executable", str(Path(temporary) / "python")):
                self.assertEqual(app._vllm_executable(), str(executable))


if __name__ == "__main__":
    unittest.main()
