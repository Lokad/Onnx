"""Synthetic end-to-end/adversarial schema-3/4 checks; no model or VM workload."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import campaign_evidence as evidence
import campaign_processes as processes
import isolated_evidence as isolated
import run_isolated_e5 as lane
import score_campaign as scorer


def save(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def make_campaign(directory, kind, schema=3):
    directory.mkdir()
    protocol = isolated.PROTOCOL if schema == 3 else isolated.CONDITIONED_PROTOCOL
    _, contract, verb = isolated.PROTOCOLS[protocol]
    duration = 10 if schema == 3 else 38
    changed = kind == "comparison"
    builds = {role: dict(source_sha=("2" if changed and role == "L1" else "1") * 40,
                        core_sha256=("4" if changed and role == "L1" else "3") * 64,
                        source_archive_sha256=("6" if changed and role == "L1" else "5") * 64, sdk="10.0.204") for role in ("L0", "L1")}
    manifest = dict(schema=schema, scope="e5", protocol=protocol, kind=kind, cooldown_seconds=300, builds=builds, oracles=[], runs=[])
    (directory / "producer").mkdir()
    manifest["producer_files"] = {}
    for name in isolated.SCRIPTS:
        path = directory / "producer" / name
        path.write_text("# synthetic producer fixture\n", encoding="utf-8")
        manifest["producer_files"][name] = isolated.binding(path, directory)
    env = dict(host="synthetic", cpu="synthetic AMD", os="synthetic Linux", architecture="x64", sdk="10.0.204", runtime=".NET 10.0.8",
               isa="AVX2,FMA", affinity="0x4", settings=dict(jit="full-opts", gc="workstation:Interactive", variables={"DOTNET_TieredCompilation": "0"}))
    native = dict(path="/synthetic/runtimes/linux-x64/native/libonnxruntime.so", sha256="7" * 64, architecture="x64")
    runner = {"runner.dll": "8" * 64}
    runner_hash = hashlib.sha256(("runner.dll\0" + "8" * 64 + "\n").encode()).hexdigest()
    origin = dt.datetime(2026, 9, 17 if changed else 16, tzinfo=dt.timezone.utc)
    offset, serial = 0, 0
    fixtures = {}

    def worker(role, mode, name):
        nonlocal offset, serial
        serial += 1
        prefix = "%03d-%s-%s-%s" % (serial, role, mode, name)
        paths = {key: directory / (prefix + suffix) for key, suffix in
                 (("process", ".process.json"), ("supervision", ".supervision.json"), ("config", ".config.json"),
                  ("log", ".log"), ("error", ".err.log"), ("before", ".pre.json"), ("after", ".post.json"))}
        fixture = fixtures[name]
        config = dict(root="/synthetic", case=name, cpu=2, output=str(paths["process"]), source_sha=builds[role]["source_sha"],
                      core_sha256=builds[role]["core_sha256"], fixture=str(fixture.parent),
                      fixture_sha256=None if mode == "oracle" else evidence.sha256(fixture), smoke=False)
        save(paths["config"], config)
        launch = origin + dt.timedelta(seconds=offset)
        before = dict(monotonic=offset, utc=(launch - dt.timedelta(seconds=1)).isoformat(), boot="synthetic", logical_cpus=4,
                      processes=[dict(id=1, parent=0, start=1, cpu=0, name="supervisor"), dict(id=99, parent=0, start=1, cpu=0, name="foreign")])
        after = copy.deepcopy(before)
        after.update(monotonic=offset + duration + 3, utc=(launch + dt.timedelta(seconds=duration + 2)).isoformat())
        save(paths["before"], before); save(paths["after"], after)
        accounting = processes.foreign_fraction(before, after, 1)
        supervision = dict(supervisor_pid=1, pid=1000 + serial, mode=mode, case=name, launched_utc=launch.isoformat(),
                           exited_utc=(launch + dt.timedelta(seconds=duration + 1)).isoformat(), exit_code=0, accounting=accounting,
                           command=["dotnet", "/synthetic/Lokad.Onnx.Campaign.dll", verb, mode, str(paths["config"])])
        save(paths["supervision"], supervision)
        paths["log"].write_text("synthetic worker\n", encoding="utf-8"); paths["error"].write_text("", encoding="utf-8")
        tick = 10000 if mode == "ort" or (changed and role == "L1") else 20000
        measured = {} if mode == "oracle" else dict(pre_scaled_error=1e-6, post_scaled_error=1e-6, first_execute_ticks=tick,
                    post_execute_ticks=tick, warmup_ticks=[200000] * 9, warmup_stop="steady",
                    blocks=[dict(index=b, block_ticks=tick * 11 + 1000, execute_ticks=[tick] * 11,
                                 allocated_bytes=1000, thread_cpu_ns=100, process_cpu_ns=100, gen0=0, gen1=0, gen2=0, gc_pause_ticks=0) for b in range(3)])
        if schema == 4 and mode != "oracle":
            measured.update(conditioning_ticks=[1000000] * 30, conditioning_wall_ticks=30100000, conditioning_stop="target-execute-30s")
        record = dict(producer=protocol, timing_contract=contract, mode=mode, case=name, smoke=False,
                      process_id=1000 + serial, started_utc=(launch + dt.timedelta(seconds=1)).isoformat(),
                      completed_utc=(launch + dt.timedelta(seconds=duration)).isoformat(), exit_code=0,
                      configuration_sha256=evidence.sha256(paths["config"]), source_sha=config["source_sha"], core_sha256=config["core_sha256"],
                      core_path="/synthetic/Lokad.Onnx.dll", runner_sha256=runner_hash, runner_files=runner, environment=env,
                      native_module=None if mode == "lok" else native, fixture_sha256=evidence.sha256(fixture),
                      model_sha256=isolated.MODEL, tokenizer_sha256="9" * 64, input_sha256=hashlib.sha256(name.encode()).hexdigest(),
                      external_data={}, unmasked_tokens=isolated.REAL[name], inputs_intact=True, ort_version="1.23.2", oracle_native_sha256=native["sha256"],
                      execution=isolated.EXECUTION, confinement=dict(wall_ms=2000, cpu_ms=2000, ratio=1), stopwatch_frequency=1000000,
                      load_ticks=100000, measured=measured)
        save(paths["process"], record)
        offset += 40
        return {key: isolated.binding(path, directory) for key, path in paths.items()}

    for name in isolated.CASES:
        directory_for_case = directory / (name + "-oracle")
        directory_for_case.mkdir()
        data = directory_for_case / "output-0.f32"
        data.write_bytes(bytes(isolated.TOKENS[name] * 384 * 4))
        fixture = dict(protocol=protocol, case=name, model_sha256=isolated.MODEL, tokenizer_sha256="9" * 64,
                       input_sha256=hashlib.sha256(name.encode()).hexdigest(), unmasked_tokens=isolated.REAL[name], oracle_version="1.23.2", native=native,
                       outputs=[dict(name="last_hidden_state", dims=[1, isolated.TOKENS[name], 384], dtype="float32", file=data.name, sha256=evidence.sha256(data))])
        fixtures[name] = directory_for_case / "fixture.json"
        save(fixtures[name], fixture)
        manifest["oracles"].append(dict(worker=worker("L0", "oracle", name), fixture=isolated.binding(fixtures[name], directory)))
    for index, role in enumerate(evidence.ORDER):
        if index > 0 and index % 2 == 0:
            offset += 300
        engines = ("lok", "ort") if index % 2 == 0 else ("ort", "lok")
        manifest["runs"].append(dict(role=role, rep=index // 2 + 1, workers=[worker(role, mode, name) for name in isolated.CASES for mode in engines]))
    save(directory / "evidence.json", manifest)
    return manifest


class IsolatedTests(unittest.TestCase):
    schema = 3
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.original = {kind: make_campaign(cls.directory / kind, kind, cls.schema) for kind in ("aa", "comparison")}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.manifests = copy.deepcopy(self.original)

    def rewrite(self, kind, binding, transform):
        path = self.directory / kind / binding["file"]
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        value = json.loads(original)
        transform(value)
        save(path, value)
        binding["sha256"] = evidence.sha256(path)
        return value

    def first(self, kind="comparison"):
        return self.manifests[kind]["runs"][0]["workers"][0]

    def change_record(self, transform, kind="comparison"):
        return self.rewrite(kind, self.first(kind)["process"], transform)

    def load(self, kind="comparison"):
        path = self.directory / kind / "evidence.json"
        save(path, self.manifests[kind])
        return evidence.load_campaign(path)

    def reject(self, text=None, kind="comparison"):
        with self.assertRaisesRegex(evidence.EvidenceError, text or "."):
            self.load(kind)

    def test_all_raw_calls_score_with_existing_policy(self):
        aa, comparison = self.load("aa"), self.load()
        result, code = scorer.score(comparison, aa)
        self.assertEqual((code, result["verdict"], result["policy"]), (0, "PASS", "amd-e5-v" + str(self.schema)))
        self.assertEqual(len(result["table"]), 5)
        self.assertEqual(result["table"][0]["L0"], 20)
        self.assertEqual(result["table"][0]["L1"], 10)
        self.assertEqual(len(comparison["runs"][0]["parsed"]["raw"]["e5-8tok"]["lok"]), 33)
        self.assertTrue(isolated.calibration_health(aa)["qualified"])

    def test_old_schema_cannot_relabel_new_worker(self):
        self.manifests["comparison"]["schema"] = 2
        self.reject()

    def test_smoke_cannot_score(self):
        self.manifests["comparison"]["kind"] = "smoke"
        self.reject("smoke cannot score")

    def test_missing_worker(self):
        self.manifests["comparison"]["runs"][0]["workers"].pop()
        self.reject("ten fresh")

    def test_missing_oracle(self):
        self.manifests["comparison"]["oracles"].pop()
        self.reject("five oracle")

    def test_reordered_worker(self):
        workers = self.manifests["comparison"]["runs"][0]["workers"]
        workers[0], workers[1] = workers[1], workers[0]
        self.reject("order/mode/case")

    def test_repeated_worker(self):
        self.manifests["comparison"]["runs"][2]["workers"] = self.manifests["comparison"]["runs"][0]["workers"]
        self.reject("overlap|order|reused")

    def test_worker_digest_binds_raw_observations(self):
        self.first()["process"]["sha256"] = "0" * 64
        self.reject("digest mismatch")

    def test_bound_paths_cannot_escape(self):
        self.first()["process"]["file"] = "../elsewhere.json"
        self.reject("relative path")

    def test_configuration_digest_is_required(self):
        self.change_record(lambda r: r.update(configuration_sha256="0" * 64))
        self.reject("configuration digest")

    def test_native_cannot_load_in_lokad_worker(self):
        self.change_record(lambda r: r.update(native_module={"path": "/native/libonnxruntime.so"}))
        self.reject("Lokad worker loaded")

    def test_native_binary_must_match_oracle(self):
        worker = self.manifests["comparison"]["runs"][0]["workers"][1]
        self.rewrite("comparison", worker["process"], lambda r: r["native_module"].update(sha256="a" * 64))
        self.reject("native module differs")

    def test_primary_call_timing_cannot_be_request_block(self):
        self.change_record(lambda r: r["measured"]["blocks"][0].update(execute_ticks=[]))
        self.reject("tick array")

    def test_block_must_enclose_its_execute_ticks(self):
        self.change_record(lambda r: r["measured"]["blocks"][0].update(block_ticks=1))
        self.reject("exceed block")

    def test_all_33_calls_required(self):
        self.change_record(lambda r: r["measured"]["blocks"][0]["execute_ticks"].pop())
        self.reject("count differs")

    def test_zero_ticks_refused(self):
        self.change_record(lambda r: r["measured"]["blocks"][0]["execute_ticks"].__setitem__(0, 0))
        self.reject("integer")

    def test_warmup_duration_required(self):
        self.change_record(lambda r: r["measured"].update(warmup_ticks=[100] * 9))
        self.reject("1000ms")

    def test_warmup_convergence_required(self):
        self.change_record(lambda r: r["measured"]["warmup_ticks"].__setitem__(8, 1000000))
        self.reject("did not converge")

    def test_post_numerical_gate(self):
        self.change_record(lambda r: r["measured"].update(post_scaled_error=0.01))
        self.reject("numerical agreement")

    def test_actual_input_integrity_required(self):
        self.change_record(lambda r: r.update(inputs_intact=False))
        self.reject("model/input contract")

    def test_native_and_managed_inputs_must_match_oracle(self):
        self.change_record(lambda r: r.update(input_sha256="0" * 64))
        self.reject("independent oracle")

    def test_runner_composite_is_recomputed(self):
        self.change_record(lambda r: r.update(runner_sha256="0" * 64))
        self.reject("runner composite")

    def test_runtime_environment_cannot_change(self):
        self.change_record(lambda r: r["environment"].update(runtime=".NET 10.0.99"))
        self.reject("host/runner/native/settings")

    def test_single_cpu_required(self):
        self.change_record(lambda r: r["environment"].update(affinity="0xC"))
        self.reject("not one CPU")

    def test_supervised_pid_required(self):
        self.change_record(lambda r: r.update(process_id=99999))
        self.reject("supervised PID")

    def test_timestamps_must_fit_supervision(self):
        self.change_record(lambda r: r.update(completed_utc="2099-01-01T00:00:00Z"))
        self.reject("supervised interval")

    def test_accounting_recomputed_from_raw_snapshots(self):
        self.rewrite("comparison", self.first()["supervision"], lambda r: r["accounting"].update(foreign_cpu_fraction=0.01))
        self.reject("accounting does not match")

    def test_true_foreign_contamination_refused(self):
        entry = self.first()
        after = self.rewrite("comparison", entry["after"], lambda r: r["processes"][1].update(cpu=20))
        before = isolated.read_json(self.directory / "comparison" / entry["before"]["file"])
        self.rewrite("comparison", entry["supervision"], lambda r: r.update(accounting=processes.foreign_fraction(before, after, 1)))
        self.reject("foreign CPU")

    def test_cooldown_contract_frozen(self):
        self.manifests["comparison"]["cooldown_seconds"] = 0
        self.reject("cooldown contract")

    def test_build_provenance_must_match_worker(self):
        self.manifests["comparison"]["builds"]["L0"]["core_sha256"] = "e" * 64
        self.reject("build provenance")

    def test_supervisor_sources_are_bound(self):
        self.manifests["comparison"]["producer_files"]["isolated_evidence.py"]["sha256"] = "0" * 64
        self.reject("digest mismatch")

    def test_bad_tick_remains_in_series_and_fails_existing_gate(self):
        def mutate(record):
            block = record["measured"]["blocks"][0]
            block["execute_ticks"][0] = 100000
            block["block_ticks"] += 80000
        self.change_record(mutate)
        result, code = scorer.score(self.load(), self.load("aa"))
        self.assertEqual((code, result["verdict"]), (3, "INCONCLUSIVE"))
        self.assertTrue(any("spread" in item for item in result["unsteady"]))

    def test_baseline_variation_cannot_qualify_comparison(self):
        def mutate(record):
            for block in record["measured"]["blocks"]:
                block["execute_ticks"] = [24000] * 11
                block["block_ticks"] = 270000
        self.change_record(mutate, "aa")
        health = isolated.calibration_health(self.load("aa"))
        self.assertFalse(health["qualified"])
        self.assertTrue(any("variation" in item for item in health["problems"]))

    def test_legacy_calibration_refused_before_staging(self):
        with mock.patch.object(lane.evidence, "load_campaign", return_value={"kind": "aa", "schema": 2}), mock.patch.object(lane.common, "stage") as stage:
            code = lane.main(["--prepared", "unused", "--output", "unused", "--cpu", "2", "--kind", "comparison", "--aa", "unused"])
        self.assertEqual(code, 2)
        stage.assert_not_called()

    def test_mixed_calibration_schema_refused_by_scorer(self):
        aa = self.load("aa")
        aa["schema"] = 2
        with self.assertRaisesRegex(evidence.EvidenceError, "schema/scope"):
            scorer.score(self.load(), aa)

    def test_oracle_output_bytes_are_verified(self):
        path = self.directory / "comparison" / "e5-8tok-oracle" / "output-0.f32"
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        path.write_bytes(b"bad")
        self.reject("oracle output digest")

    def test_missing_output_cannot_be_hidden_in_oracle_metadata(self):
        entry = self.manifests["comparison"]["oracles"][0]
        self.rewrite("comparison", entry["fixture"], lambda r: r.update(outputs=[]))
        self.rewrite("comparison", entry["worker"]["process"], lambda r: r.update(fixture_sha256=entry["fixture"]["sha256"]))
        self.reject("oracle output set")

    def test_oracle_shape_is_bound_to_canonical_case(self):
        entry = self.manifests["comparison"]["oracles"][0]
        self.rewrite("comparison", entry["fixture"], lambda r: r["outputs"][0].update(dims=[1, 4, 768]))
        self.rewrite("comparison", entry["worker"]["process"], lambda r: r.update(fixture_sha256=entry["fixture"]["sha256"]))
        self.reject("output metadata")

    def test_nonfinite_json_is_refused(self):
        self.change_record(lambda r: r["measured"].update(pre_scaled_error=float("nan")))
        self.reject("non-finite JSON")

    def test_counters_are_not_corrected_latencies(self):
        self.change_record(lambda r: r["measured"]["blocks"][0].update(thread_cpu_ns=-1))
        self.reject("integer")

    def test_simultaneous_children_refused(self):
        entry = self.manifests["comparison"]["runs"][0]["workers"][1]
        for key, fields in (("process", ("started_utc", "completed_utc")), ("supervision", ("launched_utc", "exited_utc")),
                            ("before", ("utc",)), ("after", ("utc",))):
            def shift(record, fields=fields):
                for field in fields:
                    record[field] = (evidence.timestamp(record[field]) - dt.timedelta(seconds=40)).isoformat()
            self.rewrite("comparison", entry[key], shift)
        self.reject("overlap")

    def test_actual_cooldown_cannot_be_shortened(self):
        entry = self.manifests["comparison"]["runs"][2]["workers"][0]
        for key, fields in (("process", ("started_utc", "completed_utc")), ("supervision", ("launched_utc", "exited_utc")),
                            ("before", ("utc",)), ("after", ("utc",))):
            def shift(record, fields=fields):
                for field in fields:
                    record[field] = (evidence.timestamp(record[field]) - dt.timedelta(seconds=50)).isoformat()
            self.rewrite("comparison", entry[key], shift)
        self.reject("cooldown was shortened")

    def test_oracle_preflight_detects_changed_assets_before_measurement(self):
        aa = self.load("aa")
        entry = self.manifests["comparison"]["oracles"][0]
        record = isolated.read_json(self.directory / "comparison" / entry["worker"]["process"]["file"])
        identity = copy.deepcopy(aa["signature"]["cases"][record["case"]])
        identity["tokenizer_sha256"] = "f" * 64
        with self.assertRaisesRegex(evidence.EvidenceError, "workload/oracle"):
            lane.verify_oracle_against_aa(record, identity, aa)

    def test_failed_calibration_refused_before_staging(self):
        with mock.patch.object(lane.evidence, "load_campaign", return_value={"kind": "aa", "schema": 3}), \
             mock.patch.object(lane.isolated, "calibration_health", return_value={"qualified": False, "problems": ["fixture noise"]}), \
             mock.patch.object(lane.common, "stage") as stage:
            code = lane.main(["--prepared", "unused", "--output", "unused", "--cpu", "2", "--kind", "comparison", "--aa", "unused"])
        self.assertEqual(code, 2)
        stage.assert_not_called()


class ConditionedTests(IsolatedTests):
    # Every historical semantic attack above also runs against full schema 4.
    schema = 4

    def smoke_entry(self):
        entry = self.first()
        self.rewrite("comparison", entry["config"], lambda r: r.update(smoke=True))
        def smoke(record):
            record.update(smoke=True, configuration_sha256=entry["config"]["sha256"])
            measured = record["measured"]
            measured.update(conditioning_ticks=[], conditioning_wall_ticks=0, conditioning_stop="skipped-smoke",
                            warmup_ticks=[100], warmup_stop="fixed-smoke", blocks=measured["blocks"][:1])
            measured["blocks"][0]["execute_ticks"] = [100, 100]
        self.change_record(smoke)
        return entry

    def test_smoke_explicitly_skips_conditioning(self):
        entry = self.smoke_entry()
        row = isolated.validate_process_for_protocol(self.directory / "comparison", entry, True, isolated.CONDITIONED_PROTOCOL)
        self.assertEqual(row["raw"], [0.1, 0.1])

    def test_smoke_cannot_hide_conditioning_samples(self):
        entry = self.smoke_entry()
        self.change_record(lambda r: r["measured"].update(conditioning_ticks=[1]))
        with self.assertRaisesRegex(evidence.EvidenceError, "smoke must skip conditioning"):
            isolated.validate_process_for_protocol(self.directory / "comparison", entry, True, isolated.CONDITIONED_PROTOCOL)

    def test_smoke_cannot_hide_conditioning_time(self):
        entry = self.smoke_entry()
        self.change_record(lambda r: r["measured"].update(conditioning_wall_ticks=1))
        with self.assertRaisesRegex(evidence.EvidenceError, "smoke must skip conditioning"):
            isolated.validate_process_for_protocol(self.directory / "comparison", entry, True, isolated.CONDITIONED_PROTOCOL)

    def test_full_worker_cannot_skip_conditioning(self):
        self.change_record(lambda r: r["measured"].update(conditioning_ticks=[], conditioning_wall_ticks=0, conditioning_stop="skipped-smoke"))
        self.reject("conditioning must be a nonempty")

    def test_conditioning_missing(self):
        self.change_record(lambda r: r["measured"].pop("conditioning_ticks"))
        self.reject("conditioning fields missing")

    def test_conditioning_empty(self):
        self.change_record(lambda r: r["measured"].update(conditioning_ticks=[]))
        self.reject("conditioning must be a nonempty")

    def test_conditioning_shortened(self):
        self.change_record(lambda r: r["measured"]["conditioning_ticks"].pop())
        self.reject("first reach 30s")

    def test_conditioning_continued_past_target(self):
        self.change_record(lambda r: r["measured"]["conditioning_ticks"].append(1))
        self.reject("first reach 30s")

    def test_conditioning_zero_tick(self):
        self.change_record(lambda r: r["measured"]["conditioning_ticks"].__setitem__(0, 0))
        self.reject("conditioning must be an integer")

    def test_conditioning_negative_tick(self):
        self.change_record(lambda r: r["measured"]["conditioning_ticks"].__setitem__(0, -1))
        self.reject("conditioning must be an integer")

    def test_conditioning_noninteger_tick(self):
        self.change_record(lambda r: r["measured"]["conditioning_ticks"].__setitem__(0, True))
        self.reject("conditioning must be an integer")

    def test_conditioning_nonfinite_tick(self):
        self.change_record(lambda r: r["measured"]["conditioning_ticks"].__setitem__(0, float("inf")))
        self.reject("non-finite JSON")

    def test_conditioning_call_cap(self):
        self.change_record(lambda r: r["measured"].update(conditioning_ticks=[1500] * 20001))
        self.reject("call cap exceeded")

    def test_conditioning_cannot_exceed_wall(self):
        self.change_record(lambda r: r["measured"].update(conditioning_wall_ticks=29999999))
        self.reject("wall duration/cap")

    def test_conditioning_wall_cap(self):
        self.change_record(lambda r: r["measured"].update(conditioning_wall_ticks=60000001))
        self.reject("wall duration/cap")

    def test_conditioning_stop_reason(self):
        self.change_record(lambda r: r["measured"].update(conditioning_stop="steady"))
        self.reject("conditioning stop differs")

    def test_conditioning_wall_is_in_process_closure(self):
        self.change_record(lambda r: r["measured"].update(conditioning_wall_ticks=35000000))
        self.reject("timings exceed process")

    def test_last_call_may_cross_target(self):
        self.change_record(lambda r: r["measured"]["conditioning_ticks"].__setitem__(29, 1000100))
        self.assertEqual(len(self.load()["runs"][0]["parsed"]["raw"]["e5-8tok"]["lok"]), 33)

    def test_oracle_cannot_claim_conditioning(self):
        entry = self.manifests["comparison"]["oracles"][0]["worker"]
        self.rewrite("comparison", entry["process"], lambda r: r["measured"].update(conditioning_ticks=[30000000]))
        self.reject("oracle cannot contain timed")

    def test_old_command_cannot_claim_new_protocol(self):
        self.rewrite("comparison", self.first()["supervision"], lambda r: r["command"].__setitem__(2, "isolate"))
        self.reject("supervised command differs")

    def test_old_worker_cannot_claim_new_manifest(self):
        self.change_record(lambda r: r.update(producer=isolated.PROTOCOL, timing_contract=isolated.CONTRACT))
        self.reject("worker protocol differs")

    def test_old_timing_contract_cannot_claim_conditioning(self):
        self.change_record(lambda r: r.update(timing_contract=isolated.CONTRACT))
        self.reject("worker protocol differs")

    def test_old_fixture_cannot_claim_new_protocol(self):
        entry = self.manifests["comparison"]["oracles"][0]
        self.rewrite("comparison", entry["fixture"], lambda r: r.update(protocol=isolated.PROTOCOL))
        self.rewrite("comparison", entry["worker"]["process"], lambda r: r.update(fixture_sha256=entry["fixture"]["sha256"]))
        self.reject("oracle fixture scope differs")

    def test_conditioning_cannot_be_relabeled_schema3(self):
        self.manifests["comparison"]["schema"] = 3
        self.reject("scope/protocol differs")

    def test_legacy_validator_requires_explicit_new_protocol(self):
        with self.assertRaisesRegex(evidence.EvidenceError, "worker protocol differs"):
            isolated.validate_process(self.directory / "comparison", self.first(), False)

    def test_mixed_calibration_refused_before_staging_in_both_directions(self):
        for selected, aa_schema in (("30s", 3), ("none", 4)):
            with self.subTest(selected=selected), \
                 mock.patch.object(lane.evidence, "load_campaign", return_value={"kind": "aa", "schema": aa_schema}), \
                 mock.patch.object(lane.common, "stage") as stage:
                code = lane.main(["--prepared", "unused", "--output", "unused", "--cpu", "2", "--kind", "comparison",
                                  "--aa", "unused", "--conditioning", selected])
                self.assertEqual(code, 2)
                stage.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
