"""Synthetic end-to-end and adversarial tests; no models, VM or benchmarks.

Run: python eng/test_campaign_scorer.py
"""
import contextlib
import copy
import datetime as dt
import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import campaign_evidence as evidence
import score_campaign as scorer

CASES = ("e5-8tok", "e5-30tok", "e5-30pad128", "e5-128tok", "e5-512tok",
         "dinov3-224", "resnet50-224", "gpt2-1tok", "gpt2-4tok", "gpt2-32tok",
         "gpt2-128tok", "gpt2-dec-p1", "gpt2-dec-p32", "gpt2-dec-p128", "gpt2-dec-p512")
ORDER = ("L0", "L1", "L1", "L0", "L1", "L0", "L0", "L1")


def flat(value):
    return [float(value)] * 33


def array(values):
    return "[" + ",".join(map(str, values)) + "]"


def make_log(role, kind, transform=None, drop=(), cases=CASES):
    lines = [
        "host=synthetic cpu=AMD synthetic CPU procs=4 (logical) affinity=0x4 "
        "(verified single-CPU, logical-cpu=2) topology=synthetic vector=Vector256 "
        "isa=AVX2,FMA fma=True runtime=.NET 10.0.8 lokad=" + ("0.2.1" if role == "L1" and kind == "comparison" else "0.2.0") +
        " ort=1.23.2.0 ort-provider=cpu-only ort-optimizations=ORT_ENABLE_ALL nospin "
        "intraop=1 interop=1 seq mode=auto threads=1 rows=canonical iters=33 warmup=3 warmupMin=-1 warmupMax=-1",
        "confinement wallMs=2000 cpuMs=2000 ratio=1.00 (single-threaded busy loop)"
    ]
    for name in cases:
        if name in drop:
            sequence = {"e5-8tok": 8, "e5-30tok": 30, "e5-30pad128": 128, "e5-128tok": 128, "e5-512tok": 512}.get(name)
            inputs = (",".join(key + ":1x" + str(sequence) for key in ("input_ids", "attention_mask", "token_type_ids"))
                      if sequence else "input:1x4")
            outputs = "last_hidden_state:1x%dx384" % sequence if sequence else "output:1x4"
            lines.extend([
                "case %s [single-cpu-auto-1 vs intraop=1 interop=1 seq opt=ALL nospin]: summary" % name,
                "casedef %s model=models/model.onnx bytes=1000 sha12=%s inputs=%s outputs=%s iters=33 warmup=3 tol=1.00E-004" %
                (name, "a" * 12, inputs, outputs),
                "warmup %s used=5 stop=max-reached lok=[400,400,400,400,400] ort=[400,400,400,400,400]" % name,
                "case-status %s=FAILED: synthetic quarantine" % name])
            continue
        sequence = {"e5-8tok": 8, "e5-30tok": 30, "e5-30pad128": 128, "e5-128tok": 128, "e5-512tok": 512}.get(name)
        inputs = (",".join(key + ":1x" + str(sequence) for key in ("input_ids", "attention_mask", "token_type_ids"))
                  if sequence else "input:1x4")
        outputs = "last_hidden_state:1x%dx384" % sequence if sequence else "output:1x4"
        lines.extend([
            "case %s [single-cpu-auto-1 vs intraop=1 interop=1 seq opt=ALL nospin]: summary" % name,
            "casedef %s model=models/model.onnx bytes=1000 sha12=%s inputs=%s outputs=%s iters=33 warmup=3 tol=1.00E-004" %
            (name, "a" * 12, inputs, outputs),
            "warmup %s used=3 stop=fixed lok=[400,400,400] ort=[400,400,400]" % name,
            "%s [single-cpu-auto-1 vs intraop=1 interop=1 seq opt=ALL nospin]: maxScaled=1E-6 postScaled=1E-6 inputsIntact=yes" % name
        ])
        lok = flat(10 if kind == "comparison" and role == "L1" and name.startswith("e5-") else 20)
        ort = flat(10)
        if transform:
            lok, ort = transform(name, lok, ort)
        lines.append("raw lok=%s raw ctx=%s raw ort=%s" % (array(lok), array(lok), array(ort)))
        lines.append("case-status %s=ok" % name)
    return "\n".join(lines) + "\n"


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.aa = self.make_campaign("aa")
        self.candidate = self.make_campaign("comparison")

    def make_campaign(self, kind, drop=(), scope=None):
        cases = CASES if scope != "e5" else CASES[:5]
        manifest = {"schema": 1 if scope is None else 2, "kind": kind, "runs": []}
        if scope is not None:
            manifest["scope"] = scope
        origin = dt.datetime(2026, 9, 16 if kind == "aa" else 17, tzinfo=dt.timezone.utc)
        for i, role in enumerate(ORDER):
            log = self.directory / (kind + "-%d.log" % i)
            log.write_text(make_log(role, kind, drop=drop, cases=cases), encoding="utf-8")
            changed = role == "L1" and kind == "comparison"
            start = origin + dt.timedelta(minutes=2 * i)
            manifest["runs"].append({
                "role": role, "rep": i // 2 + 1, "log": log.name, "log_sha256": evidence.sha256(log),
                "process_id": 10000 + i, "started_utc": start.isoformat(),
                "completed_utc": (start + dt.timedelta(seconds=100)).isoformat(), "exit_code": 0,
                "source_sha": ("2" if changed else "1") * 40, "core_sha256": ("4" if changed else "3") * 64,
                "runner_sha256": "5" * 64,
                "ort_native": {"path": "/synthetic/runtimes/linux-x64/native/libonnxruntime.so", "sha256": "6" * 64, "architecture": "x64"},
                "environment": {"host": "synthetic", "cpu": "AMD synthetic CPU", "os": "Linux synthetic",
                                "architecture": "x64", "sdk": "10.0.204", "runtime": ".NET 10.0.8", "isa": "AVX2,FMA", "affinity": "0x4",
                                "settings": {"jit": "default-tiered", "gc": "workstation", "variables": {}}},
                "accounting": {"valid": True, "foreign_cpu_fraction": 0.0},
                "cases": {name: {"model_sha256": "a" * 64, "input_sha256": "b" * 64, "external_data": {},
                                 **({"unmasked_tokens": {"e5-8tok": 8, "e5-30tok": 30, "e5-30pad128": 30, "e5-128tok": 128, "e5-512tok": 512}[name]}
                                    if name.startswith("e5-") else {})} for name in cases},
                "cases_failed": sorted(drop, key=CASES.index)
            })
            if scope is not None:
                run = manifest["runs"][-1]
                run.update(producer="common-runner-v2", scope=scope, runner_files={"runner.dll": "5" * 64})
                run["runner_sha256"] = hashlib.sha256(("runner.dll\0" + "5" * 64 + "\n").encode()).hexdigest()
                self.bind_process(run)
        return manifest

    def bind_process(self, run):
        path = self.directory / (run["log"] + ".process.json")
        path.write_text(json.dumps(run), encoding="utf-8")
        run.update(process_evidence=path.name, process_evidence_sha256=evidence.sha256(path))

    def rewrite_log(self, manifest, index, transform):
        run = manifest["runs"][index]
        path = self.directory / run["log"]
        path.write_text(transform(path.read_text(encoding="utf-8")), encoding="utf-8")
        run["log_sha256"] = evidence.sha256(path)

    def change_samples(self, manifest, indexes, transform):
        for i in indexes:
            run = manifest["runs"][i]
            self.rewrite_log(manifest, i, lambda _: make_log(run["role"], manifest["kind"], transform))

    def evaluate(self, expected_code=0, verdict=None):
        paths = []
        for name, manifest in (("candidate", self.candidate), ("aa", self.aa)):
            path = self.directory / (name + ".json")
            path.write_text(json.dumps(manifest), encoding="utf-8")
            paths.append(path)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = scorer.main(["--evidence", str(paths[0]), "--aa", str(paths[1])])
        text = output.getvalue()
        self.assertEqual(code, expected_code, text)
        if verdict:
            result = json.loads(text[text.rfind("\n{") + 1:])
            self.assertEqual(result["verdict"], verdict, text)
            return result
        return text

    def test_parity_without_resnet_improvement(self):
        result = self.evaluate(verdict="PASS")
        self.assertEqual(set(result["targets"]), {"e5-8tok", "e5-30tok", "e5-128tok", "e5-30pad128"})
        self.assertAlmostEqual(result["e5_improvement"], 0.5)
        self.assertNotIn("family_score", result)
        self.assertEqual(len(result["table"]), 15)
        e5 = result["table"][0]
        self.assertEqual((e5["L0"], e5["L1"], e5["O0"], e5["O1"], e5["R0"], e5["R1"], e5["gap_closure"]), (20, 10, 10, 10, 2, 1, 1))
        self.assertEqual(len(e5["reps"]), 4)

    def scoped_campaigns(self, scope="e5", drop=()):
        self.aa = self.make_campaign("aa", scope=scope)
        self.candidate = self.make_campaign("comparison", scope=scope, drop=drop)

    def test_e5_scope_scores_five_ordered_cases(self):
        self.scoped_campaigns()
        result = self.evaluate(verdict="PASS")
        self.assertEqual(result["scope"], "e5")
        self.assertEqual(result["policy"], "amd-e5-v2")
        self.assertEqual(tuple(row["case"] for row in result["table"]), CASES[:5])

    def test_full_schema_two_scores_all_cases(self):
        self.scoped_campaigns("full")
        self.assertEqual(len(self.evaluate(verdict="PASS")["table"]), 15)

    def test_e5_scope_allows_explicit_long_quarantine(self):
        self.scoped_campaigns(drop=("e5-512tok",))
        self.assertEqual(self.evaluate(verdict="PASS")["quarantined"], {"e5-512tok": ["comparison"]})

    def test_e5_scope_primary_quarantine_is_inconclusive(self):
        self.scoped_campaigns(drop=("e5-30tok",))
        self.evaluate(expected_code=3, verdict="INCONCLUSIVE")

    def test_e5_scope_missing_long_case_is_rejected(self):
        self.scoped_campaigns()
        run = self.candidate["runs"][0]
        del run["cases"]["e5-512tok"]
        self.bind_process(run)
        self.assertIn("manifest case identities missing/extra", self.evaluate(expected_code=2))

    def test_e5_scope_truncated_log_is_rejected(self):
        self.scoped_campaigns()
        self.rewrite_log(self.candidate, 0, lambda _: make_log("L0", "comparison", cases=CASES[:4]))
        self.assertIn("case set mismatch", self.evaluate(expected_code=2))

    def test_scope_is_required_in_schema_two(self):
        self.scoped_campaigns()
        del self.candidate["scope"]
        self.evaluate(expected_code=2)

    def test_unknown_scope_is_rejected(self):
        self.scoped_campaigns()
        self.candidate["scope"] = "e5-fast-only"
        self.assertIn("unsupported evidence scope", self.evaluate(expected_code=2))

    def test_legacy_evidence_cannot_be_relabelled(self):
        self.candidate.update(schema=2, scope="full")
        self.assertIn("scope-bound process evidence", self.evaluate(expected_code=2))

    def test_legacy_schema_cannot_declare_scope(self):
        self.candidate["scope"] = "full"
        self.assertIn("legacy evidence cannot declare", self.evaluate(expected_code=2))

    def test_mixed_scopes_between_runs_are_rejected(self):
        self.scoped_campaigns()
        self.candidate["runs"][1]["scope"] = "full"
        self.assertIn("scope differs from campaign", self.evaluate(expected_code=2))

    def test_manifest_scope_is_bound_to_original_process(self):
        self.scoped_campaigns("full")
        self.candidate["scope"] = "e5"
        for run in self.candidate["runs"]:
            run["scope"] = "e5"
        self.assertIn("process evidence/manifest mismatch: scope", self.evaluate(expected_code=2))

    def test_mixed_aa_scopes_are_rejected(self):
        self.scoped_campaigns()
        self.aa = self.make_campaign("aa", scope="full")
        self.assertIn("A/A evidence schema/scope mismatch", self.evaluate(expected_code=2))

    def test_schema_two_cannot_reuse_legacy_calibration(self):
        self.candidate = self.make_campaign("comparison", scope="full")
        self.assertIn("A/A evidence schema/scope mismatch", self.evaluate(expected_code=2))

    def test_scoped_quarantine_does_not_hide_changed_input(self):
        self.scoped_campaigns(drop=("e5-512tok",))
        for run in self.candidate["runs"]:
            run["cases"]["e5-512tok"]["input_sha256"] = "c" * 64
            self.bind_process(run)
        self.assertIn("including quarantined cases", self.evaluate(expected_code=2))

    def test_forged_scoped_manifest_identity_is_rejected(self):
        self.scoped_campaigns()
        self.candidate["runs"][0]["cases"]["e5-8tok"]["input_sha256"] = "c" * 64
        self.assertIn("process evidence/manifest mismatch: cases", self.evaluate(expected_code=2))

    def test_quarantined_case_is_inconclusive_not_regression(self):
        self.candidate = self.make_campaign("comparison", drop=("resnet50-224",))
        self.aa = self.make_campaign("aa", drop=("dinov3-224",))
        result = self.evaluate(verdict="PASS")
        self.assertEqual(result["quarantined"], {"dinov3-224": ["aa"], "resnet50-224": ["comparison"]})
        self.assertEqual(len(result["table"]), 15)
        marked = [row for row in result["table"] if "quarantined" in row]
        self.assertEqual({row["case"] for row in marked}, {"dinov3-224", "resnet50-224"})
        self.assertNotIn("resnet50-224", result.get("regressions", []))
        self.assertNotIn("dinov3-224", result.get("regressions", []))

    def test_wall_cap_stop_quarantines_like_max_reached(self):
        self.candidate = self.make_campaign("comparison", drop=("resnet50-224",))
        run = self.candidate["runs"][0]
        path = self.directory / run["log"]
        text = path.read_text(encoding="utf-8")
        old = "warmup resnet50-224 used=5 stop=max-reached"
        self.assertIn(old, text)
        path.write_text(text.replace(old, "warmup resnet50-224 used=5 stop=wall-cap"), encoding="utf-8")
        run["log_sha256"] = evidence.sha256(path)
        result = self.evaluate(verdict="PASS")
        self.assertEqual(result["quarantined"], {"resnet50-224": ["comparison"]})
        marked = [row for row in result["table"] if "quarantined" in row]
        self.assertEqual({row["case"] for row in marked}, {"resnet50-224"})

    def test_quarantined_primary_case_is_inconclusive(self):
        self.candidate = self.make_campaign("comparison", drop=("e5-30tok",))
        result = self.evaluate(expected_code=3, verdict="INCONCLUSIVE")
        self.assertIn("e5-30tok", result["quarantined"])
        self.assertIsNone(result.get("e5_improvement"))

    def test_quarantine_disagreement_across_reps_is_union_not_error(self):
        for index in (0, 1):
            run = self.candidate["runs"][index]
            (self.directory / run["log"]).write_text(make_log(run["role"], "comparison", drop=("dinov3-224",)), encoding="utf-8")
            run["log_sha256"] = evidence.sha256(self.directory / run["log"])
            run["cases_failed"] = ["dinov3-224"]
        result = self.evaluate(verdict="PASS")
        self.assertEqual(result["quarantined"], {"dinov3-224": ["comparison"]})
        self.assertEqual(len(result["table"]), 15)

    def test_parity_miss_is_not_regression(self):
        self.change_samples(self.candidate, [1, 2, 4, 7], lambda name, lok, ort: (flat(15) if name.startswith("e5-") else lok, ort))
        self.evaluate(verdict="MISS")

    def test_padded_long_and_decode_regressions_are_checked(self):
        for name in ("e5-30pad128", "e5-512tok", "gpt2-dec-p512"):
            with self.subTest(name=name):
                self.change_samples(self.candidate, [1, 2, 4, 7], lambda case, lok, ort: (flat(21) if case == name else lok, ort))
                result = self.evaluate(verdict="REGRESSION")
                self.assertEqual(result["regressions"], [name])

    def test_chronological_drift_not_sorted_quantiles(self):
        # Same histogram: temporal ordering alone decides the half-split gate.
        alternating = [9.4, 10.6] * 16 + [10.0]
        drifting = sorted(alternating)
        self.assertEqual(scorer.steady("alternating", alternating), "")
        self.assertIn("half-split", scorer.steady("drifting", drifting))
        self.change_samples(self.candidate, [1, 2, 4, 7], lambda name, lok, ort: (alternating if name == "e5-8tok" else lok, ort))
        self.evaluate(verdict="PASS")
        self.change_samples(self.candidate, [1, 2, 4, 7], lambda name, lok, ort: (drifting if name == "e5-8tok" else lok, ort))
        self.evaluate(3, "INCONCLUSIVE")

    def test_candidate_noise_cannot_widen_regression_band(self):
        self.change_samples(self.candidate, [1], lambda name, lok, ort: (flat(22), ort))
        result = self.evaluate(3, "INCONCLUSIVE")
        self.assertEqual(result["noise"]["e5-8tok"]["regression_limit"], 0.03)
        self.assertNotIn("table", result)

    def test_noisy_aa_is_inconclusive(self):
        self.change_samples(self.aa, [0], lambda name, lok, ort: (flat(21), ort))
        self.evaluate(3, "INCONCLUSIVE")

    def test_frozen_aa_allowance_is_used(self):
        self.change_samples(self.aa, [0, 3, 5, 6], lambda name, lok, ort: (flat(19.8), ort))
        self.change_samples(self.aa, [1, 2, 4, 7], lambda name, lok, ort: (flat(20.2), ort))
        self.change_samples(self.candidate, [1, 2, 4, 7], lambda name, lok, ort: (flat(20.7) if name == "resnet50-224" else lok, ort))
        result = self.evaluate(verdict="PASS")
        self.assertAlmostEqual(result["noise"]["resnet50-224"]["regression_limit"], 0.04)

    def test_ort_controls_not_pooled(self):
        self.change_samples(self.candidate, [1, 2, 4, 7], lambda name, lok, ort: (lok, flat(12)))
        result = self.evaluate(3, "INCONCLUSIVE")
        self.assertTrue(any("ORT control drift" in item for item in result["unsteady"]))

    def test_small_ort_difference_is_reported_separately(self):
        self.change_samples(self.candidate, [1, 2, 4, 7], lambda name, lok, ort: (lok, flat(10.2)))
        result = self.evaluate(verdict="PASS")
        row = result["table"][0]
        self.assertEqual((row["O0"], row["O1"]), (10, 10.2))
        self.assertAlmostEqual(row["R1"], 10 / 10.2)

    def test_wrong_mask_or_shapes_cannot_be_consistent_across_all_runs(self):
        for manifest in (self.aa, self.candidate):
            for run in manifest["runs"]:
                run["cases"]["e5-30pad128"]["unmasked_tokens"] = 128
        self.assertIn("attention mask", self.evaluate(2))
        for manifest in (self.aa, self.candidate):
            for index, run in enumerate(manifest["runs"]):
                run["cases"]["e5-30pad128"]["unmasked_tokens"] = 30
                self.rewrite_log(manifest, index, lambda text: text.replace("inputs=input_ids:1x8", "inputs=input_ids:1x7"))
        self.assertIn("input shape", self.evaluate(2))

    def test_full_opts_is_labelled_and_requires_matching_aa(self):
        for run in self.candidate["runs"]:
            run["environment"]["settings"].update(jit="full-opts", variables={"DOTNET_TieredCompilation": "0"})
        self.assertIn("A/A workload", self.evaluate(2))
        for run in self.aa["runs"]:
            run["environment"]["settings"].update(jit="full-opts", variables={"DOTNET_TieredCompilation": "0"})
        self.assertEqual(self.evaluate(verdict="PASS")["jit_regime"], "full-opts")

    def test_missing_long_case(self):
        self.rewrite_log(self.candidate, 0, lambda text: text.replace("case-status e5-512tok=ok\n", ""))
        self.assertIn("case set mismatch", self.evaluate(2))

    def test_runtime_patch_is_compared(self):
        self.rewrite_log(self.candidate, 0, lambda text: text.replace(".NET 10.0.8", ".NET 10.0.9"))
        self.assertIn("runtime", self.evaluate(2))

    def test_changed_input_native_runner_and_core(self):
        original = copy.deepcopy(self.candidate)
        changes = (
            lambda r: r["cases"]["e5-30pad128"].update(input_sha256="c" * 64),
            lambda r: r["cases"]["dinov3-224"].update(external_data={"model.onnx_data": "c" * 64}),
            lambda r: r["ort_native"].update(sha256="c" * 64),
            lambda r: r.update(runner_sha256="c" * 64),
            lambda r: r.update(core_sha256="c" * 64),
        )
        for change in changes:
            with self.subTest(change=change):
                self.candidate = copy.deepcopy(original)
                change(self.candidate["runs"][1])
                self.evaluate(2)

    def test_failed_evidence_metadata(self):
        original = copy.deepcopy(self.candidate)
        changes = (
            lambda r: r.update(exit_code=1),
            lambda r: r["accounting"].update(valid=False),
            lambda r: r["accounting"].update(foreign_cpu_fraction=0.11),
            lambda r: r["accounting"].update(foreign_cpu_fraction=float("nan")),
            lambda r: r["ort_native"].update(path="/runtimes/android/native/onnxruntime.aar"),
            lambda r: r["ort_native"].update(path="libonnxruntime.so"),
            lambda r: r["ort_native"].update(architecture="arm64"),
            lambda r: r["ort_native"].update(path="/runtimes/linux-arm64/native/libonnxruntime.so"),
            lambda r: r["environment"]["settings"].update(jit="full-opts"),
            lambda r: r["environment"]["settings"].update(variables={"COMPlus_TieredCompilation": "0"}),
            lambda r: r.update(source_sha="origin/master"),
            lambda r: r.update(completed_utc=r["started_utc"]),
            lambda r: r.update(rep=True),
        )
        for change in changes:
            with self.subTest(change=change):
                self.candidate = copy.deepcopy(original)
                change(self.candidate["runs"][0])
                self.evaluate(2)

    def test_failed_log_evidence(self):
        run = self.candidate["runs"][0]
        original = (self.directory / run["log"]).read_text(encoding="utf-8")
        changes = (
            ("affinity=0x4", "affinity=0xC"),
            ("threads=1", "threads=2"),
            ("cpuMs=2000 ratio=1.00", "cpuMs=4000 ratio=2.00"),
            ("inputsIntact=yes", "inputsIntact=no"),
            ("maxScaled=1E-6", "maxScaled=0.1"),
            ("postScaled=1E-6", "postScaled=NaN"),
            ("warmup e5-8tok used=3", "warmup e5-8tok used=2"),
            ("ort=[400,400,400]", "ort=[10,10,10]"),
            ("inputs=input_ids:1x8", "inputs=input_ids:1x7"),
            ("raw lok=[20.0,", "raw lok=[NaN,"),
            ("raw lok=[20.0,", "raw lok=[0,"),
            ("raw lok=[20.0,", "raw lok=[-1,"),
            ("raw lok=[20.0,", "raw lok=[1e308,"),
            ("raw ort=[10.0,", "raw ort=["),
            ("case-status e5-8tok=ok", "case-status e5-8tok=ok\ncase-status e5-8tok=ok"),
            ("case-status e5-8tok=ok", "case-status e5-8tok=excluded-known-divergence"),
        )
        for before, after in changes:
            with self.subTest(after=after):
                self.rewrite_log(self.candidate, 0, lambda _: original.replace(before, after))
                self.evaluate(2)

    def test_log_digest_binding(self):
        path = self.directory / self.candidate["runs"][0]["log"]
        path.write_text(path.read_text() + "changed\n", encoding="utf-8")
        self.assertIn("log digest mismatch", self.evaluate(2))

    def test_repeated_log_and_wrong_order(self):
        self.candidate["runs"][1]["log"] = self.candidate["runs"][0]["log"]
        self.assertIn("reused log", self.evaluate(2))
        self.candidate["runs"].reverse()
        self.assertIn("order", self.evaluate(2))

    def test_fewer_than_four_pairs(self):
        self.candidate["runs"] = self.candidate["runs"][:6]
        self.assertIn("four paired", self.evaluate(2))

    def test_aa_cannot_be_a_candidate_comparison(self):
        self.aa["runs"][1]["core_sha256"] = "4" * 64
        self.evaluate(2)

    def test_aa_chronology(self):
        for run in self.aa["runs"]:
            run["started_utc"] = run["started_utc"].replace("09-16", "09-18")
            run["completed_utc"] = run["completed_utc"].replace("09-16", "09-18")
        self.assertIn("A/A must finish before", self.evaluate(2))

    def test_aa_workload_identity(self):
        for run in self.aa["runs"]:
            run["cases"]["e5-8tok"]["input_sha256"] = "c" * 64
        self.assertIn("A/A workload", self.evaluate(2))

    def test_utf16_log_supported(self):
        run = self.candidate["runs"][0]
        path = self.directory / run["log"]
        path.write_text(path.read_text(), encoding="utf-16")
        run["log_sha256"] = evidence.sha256(path)
        self.evaluate(verdict="PASS")

    def test_cli_and_old_invocation(self):
        self.evaluate(verdict="PASS")
        script = Path(__file__).with_name("score_campaign.py")
        proc = subprocess.run([sys.executable, str(script), "--evidence", str(self.directory / "candidate.json"),
                               "--aa", str(self.directory / "aa.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn('"verdict": "PASS"', proc.stdout)
        proc = subprocess.run([sys.executable, str(script), "old0.log", "--l1", "old1.log"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("requires --evidence", proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)

    def test_malformed_manifest_is_actionable(self):
        path = self.directory / "bad.json"
        for content in ('{}', '{"schema":1,"schema":1}', '{', 'null', '[]', '{"schema":NaN}'):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(evidence.EvidenceError):
                    evidence.load_campaign(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
