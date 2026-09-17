"""Offline supervision/accounting tests; no VM or CPU burn workloads."""
import copy
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import subprocess
import sys

import campaign_evidence as evidence
import campaign_processes as processes
import run_common_campaign as lane


def row(pid, parent=0, start=10, cpu=0):
    return {"id": pid, "parent": parent, "start": start, "cpu": cpu, "name": "fixture"}


def snapshot(rows, when):
    return {"processes": rows, "monotonic": when, "boot": "fixture-boot", "logical_cpus": 4}


class AccountingTests(unittest.TestCase):
    def test_linux_ticks_and_names_with_parentheses(self):
        fields = ["S", "42"] + ["0"] * 18
        fields[11], fields[12], fields[19] = "1200", "300", "999"
        parsed = processes.linux_row("73 (a ) tricky name) " + " ".join(fields), 100)
        self.assertEqual((parsed["id"], parsed["parent"], parsed["cpu"], parsed["start"]), (73, 42, 15, 999))
        self.assertEqual(parsed["name"], "a ) tricky name")

    def test_bad_linux_rows_fail_instead_of_zero(self):
        for line in ("", "12 bad", "12 (x) S 4 0"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                processes.linux_row(line, 100)

    def test_invalid_snapshot_rows_fail_closed(self):
        for rows in ([], [row(1), row(1)], [row(1, cpu=float("nan"))], [row(1, cpu=-1)],
                     [dict(row(1), start=None)], [row(-1)], [row(1, parent=-1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                processes.checked_rows(rows)

    def test_owned_children_excluded_foreign_burn_detected(self):
        before = snapshot([row(1), row(2, 1, 11), row(3, 2, 12), row(99, cpu=7)], 100)
        after = snapshot([row(1, cpu=10), row(2, 1, 11, 20), row(3, 2, 12, 30), row(99, cpu=17)], 110)
        result = processes.foreign_fraction(before, after, 1)
        self.assertEqual(result["foreign_cpu_fraction"], 0.25)
        self.assertEqual(result["foreign_cpu_seconds"], 10)

    def test_pid_reuse_counts_new_lifetime(self):
        before = snapshot([row(1), row(99, cpu=100)], 100)
        after = snapshot([row(1), row(99, start=20, cpu=8)], 110)
        result = processes.foreign_fraction(before, after, 1)
        self.assertEqual(result["foreign_cpu_fraction"], 0.2)
        self.assertEqual(result["exited_foreign_processes"], 1)

    def test_parent_reuse_and_cycles_cannot_claim_ownership(self):
        table = processes.checked_rows([row(1, start=100), row(2, 1, 50), row(3, 4), row(4, 3)])
        self.assertFalse(processes.owned(2, table, (1, 100)))
        self.assertFalse(processes.owned(3, table, (1, 100)))
        self.assertFalse(processes.owned(1, table, (1, 99)))

    def test_regressing_counters_and_bad_intervals_fail(self):
        before = snapshot([row(1), row(99, cpu=100)], 100)
        candidates = [snapshot([row(1), row(99, cpu=99)], 110),
                      snapshot([row(1, start=20), row(99, cpu=100)], 110),
                      snapshot([row(1), row(99, cpu=100)], 100)]
        for after in candidates:
            with self.subTest(after=after), self.assertRaises(ValueError):
                processes.foreign_fraction(before, after, 1)

    def test_exited_process_limitation_is_reported(self):
        result = processes.foreign_fraction(snapshot([row(1), row(99)], 100), snapshot([row(1)], 110), 1)
        self.assertEqual(result["exited_foreign_processes"], 1)
        self.assertIn("exited", result["limitation"])


class LaunchTests(unittest.TestCase):
    def test_linux_pins_before_exec_without_shell(self):
        with mock.patch.object(lane.sys, "platform", "linux"), mock.patch.object(lane.subprocess, "Popen") as launch:
            lane.spawn_child(["dotnet", "runner.dll"], 4, cwd="fixture")
        launch.assert_called_once_with(["taskset", "--cpu-list", "4", "dotnet", "runner.dll"], cwd="fixture")

    @unittest.skipUnless(sys.platform == "win32", "Windows affinity inheritance")
    def test_windows_child_inherits_pin_and_parent_is_restored(self):
        original = lane.windows_affinity()[2]
        cpu = (original & -original).bit_length() - 1
        command = [sys.executable, "-c", "import run_common_campaign as lane; print(lane.windows_affinity()[2])"]
        child = lane.spawn_child(command, cpu, cwd=Path(lane.__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = child.communicate(timeout=20)
            self.assertEqual(child.returncode, 0, stderr)
            self.assertEqual(int(stdout), 1 << cpu)
            self.assertEqual(lane.windows_affinity()[2], original)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    @unittest.skipUnless(sys.platform == "win32", "Windows affinity restoration")
    def test_failed_windows_launch_restores_parent(self):
        original = lane.windows_affinity()[2]
        cpu = (original & -original).bit_length() - 1
        with mock.patch.object(lane.subprocess, "Popen", side_effect=OSError("fixture launch failure")):
            with self.assertRaisesRegex(OSError, "fixture launch failure"):
                lane.spawn_child(["unused"], cpu)
        self.assertEqual(lane.windows_affinity()[2], original)


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        (self.prepared / "runner").mkdir(parents=True)
        for name, content in ((lane.RUNNER, b"common runner"), ("shared.dll", b"same"), ("Lokad.Onnx.dll", b"L0")):
            (self.prepared / "runner" / name).write_bytes(content)
        for role in ("L0", "L1"):
            leg = self.prepared / role
            (leg / "core").mkdir(parents=True)
            core = leg / "core" / "Lokad.Onnx.dll"
            core.write_bytes(role.encode())
            archive = leg / "source.zip"
            archive.write_bytes(b"synthetic archive, never executed")
            lane.save_json(leg / "build.json", {"core": "core/Lokad.Onnx.dll", "core_sha256": evidence.sha256(core),
                           "source_sha": ("a" if role == "L0" else "b") * 40, "sdk": "10.0.204",
                           "source_archive_sha256": evidence.sha256(archive)})

    def test_staging_only_changes_core(self):
        staged = lane.stage(self.prepared, self.root / "run", "comparison")
        self.assertEqual(staged["L0"]["runner"].read_bytes(), staged["L1"]["runner"].read_bytes())
        self.assertEqual((staged["L1"]["runner"].parent / "shared.dll").read_bytes(), b"same")
        self.assertEqual((staged["L0"]["runner"].parent / "Lokad.Onnx.dll").read_bytes(), b"L0")
        self.assertEqual((staged["L1"]["runner"].parent / "Lokad.Onnx.dll").read_bytes(), b"L1")
        self.assertEqual((self.prepared / "runner" / "Lokad.Onnx.dll").read_bytes(), b"L0")

    def test_aa_stages_identical_baseline(self):
        staged = lane.stage(self.prepared, self.root / "run", "aa")
        self.assertEqual(staged["L0"]["core_sha256"], staged["L1"]["core_sha256"])
        self.assertEqual((staged["L1"]["runner"].parent / "Lokad.Onnx.dll").read_bytes(), b"L0")

    def test_existing_or_recursive_output_is_refused(self):
        for output in (self.prepared, self.prepared / "nested"):
            with self.subTest(output=output), self.assertRaises(evidence.EvidenceError):
                lane.stage(self.prepared, output, "aa")

    def test_modified_core_or_archive_is_refused(self):
        (self.prepared / "L0" / "core" / "Lokad.Onnx.dll").write_bytes(b"changed")
        with self.assertRaisesRegex(evidence.EvidenceError, "core changed"):
            lane.read_build(self.prepared, "L0")
        (self.prepared / "L1" / "source.zip").write_bytes(b"changed")
        with self.assertRaisesRegex(evidence.EvidenceError, "archive changed"):
            lane.read_build(self.prepared, "L1")

    def test_finish_binds_observed_exit_pid_and_log(self):
        build = lane.read_build(self.prepared, "L0")
        start = dt.datetime(2026, 9, 17, tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(seconds=10)
        record = {"producer": "common-runner-v1", "process_id": 42, "exit_code": 0,
                  "started_utc": (start + dt.timedelta(seconds=1)).isoformat(), "completed_utc": (end - dt.timedelta(seconds=1)).isoformat(),
                  "core_sha256": build["core_sha256"], "source_sha": build["source_sha"], "environment": {"sdk": build["sdk"]}}
        path, log = self.root / "process.json", self.root / "child.log"
        lane.save_json(path, record)
        log.write_text("real bytes to bind\n")
        finished = lane.finish_record(path, build, "L0", 1, 42, start, end, 0, log, {"valid": True})
        self.assertEqual(finished["log_sha256"], evidence.sha256(log))
        self.assertEqual(finished["process_evidence_sha256"], evidence.sha256(path))
        for code, pid in ((1, 42), (0, 99)):
            with self.subTest(code=code, pid=pid), self.assertRaises(evidence.EvidenceError):
                lane.finish_record(path, build, "L0", 1, pid, start, end, code, log, {"valid": True})
        changed = dict(build, core_sha256="c" * 64)
        with self.assertRaises(evidence.EvidenceError):
            lane.finish_record(path, changed, "L0", 1, 42, start, end, 0, log, {"valid": True})

    def test_missing_process_evidence_is_not_success(self):
        start = dt.datetime.now(dt.timezone.utc)
        with self.assertRaises(FileNotFoundError):
            lane.finish_record(self.root / "missing.json", {}, "L0", 1, 42, start, start, 0, self.root / "x.log", {})

    def test_old_evidence_cannot_be_overwritten(self):
        path = self.root / "evidence.json"
        lane.save_json(path, {"original": True})
        with self.assertRaises(FileExistsError):
            lane.save_json(path, {"original": False})

    def test_process_observations_and_runner_bundle_are_bound(self):
        import hashlib
        files = {"app.dll": "a" * 64}
        composite = hashlib.sha256(("app.dll\0" + "a" * 64 + "\n").encode()).hexdigest()
        observed = {"producer": "common-runner-v1", "process_id": 42,
                    "started_utc": "2026-09-17T00:00:00Z", "completed_utc": "2026-09-17T00:01:00Z",
                    "exit_code": 0, "source_sha": "b" * 40, "core_sha256": "c" * 64,
                    "runner_sha256": composite, "runner_files": files, "ort_native": {}, "environment": {}, "cases": {},
                    "cases_failed": []}
        path = self.root / "observations.json"
        lane.save_json(path, observed)
        run = dict(observed, process_evidence=path.name, process_evidence_sha256=evidence.sha256(path))
        evidence.check_process_evidence(run, self.root)
        changed = dict(run, core_sha256="d" * 64)
        with self.assertRaisesRegex(evidence.EvidenceError, "manifest mismatch"):
            evidence.check_process_evidence(changed, self.root)
        changed = dict(run, cases_failed=["resnet50-224"])
        with self.assertRaisesRegex(evidence.EvidenceError, "manifest mismatch"):
            evidence.check_process_evidence(changed, self.root)
        path.write_text(json.dumps(dict(observed, core_sha256="d" * 64)))
        with self.assertRaisesRegex(evidence.EvidenceError, "digest mismatch"):
            evidence.check_process_evidence(run, self.root)
        observed["runner_sha256"] = "0" * 64
        path.write_text(json.dumps(observed))
        run = dict(observed, process_evidence=path.name, process_evidence_sha256=evidence.sha256(path))
        with self.assertRaisesRegex(evidence.EvidenceError, "composite digest"):
            evidence.check_process_evidence(run, self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
