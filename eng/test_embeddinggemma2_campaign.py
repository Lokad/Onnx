"""Synthetic protocol refusal checks; no model weights or inference timings."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import campaign_evidence as evidence
import campaign_processes as processes
import isolated_evidence as isolated
import run_embeddinggemma2 as lane


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_smoke(root):
    root.mkdir()
    for role in ("L0", "L1"):
        binary = root / "staged" / role / "Lokad.Onnx.dll"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"synthetic core")
    core = evidence.sha256(root / "staged/L0/Lokad.Onnx.dll")
    oracle = dict(schemaVersion=1, protocol=lane.PROTOCOL, modelSha256=lane.MODEL, weightsSha256=lane.WEIGHTS,
        tokenizerSha256=lane.TOKENIZER, preparedGraphSha256="a" * 64, preparationSha256="b" * 64, cases=[])
    for name in lane.CASES:
        width, real = lane.WIDTHS[name], lane.REAL[name]
        case = dict(id=name, sequenceLength=width, unmaskedTokens=real)
        directory = root / "fixtures" / name
        directory.mkdir(parents=True)
        arrays = {"input_ids": ("int64", [2] + [10] * (real - 2) + [1] + [0] * (width - real)),
                  "attention_mask": ("int64", [1] * real + [0] * (width - real))}
        arrays.update({f"embedding_{d}": ("float32", [1.] + [0.] * (d - 1)) for d in (128, 256, 512, 768)})
        for key, (dtype, values) in arrays.items():
            path = directory / key
            path.write_bytes(struct.pack("<" + ("q" if dtype == "int64" else "f") * len(values), *values))
            case[key] = dict(file=key, dtype=dtype, shape=[1, len(values)], sha256=evidence.sha256(path))
        oracle["cases"].append(case)
    save(root / "fixtures/fixtures.json", oracle)
    fixture_hash = evidence.sha256(root / "fixtures/fixtures.json")
    producers = []
    for name in lane.SCRIPTS:
        path = root / "producers" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# synthetic producer\n")
        producers.append(dict(name=name, binding=isolated.binding(path, root)))
    builds = {}
    for role in ("L0", "L1"):
        archive = root / "builds" / role / "source.zip"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"synthetic source")
        path = archive.with_name("build.json")
        save(path, dict(source_sha="1" * 40, core_sha256=core, source_archive_sha256=evidence.sha256(archive), sdk="10.0.204"))
        builds[role] = dict(build=isolated.binding(path, root), source_archive=isolated.binding(archive, root))
    env = dict(host="synthetic", cpu="synthetic AMD", os="synthetic Linux", architecture="x64", sdk="10.0.204",
        runtime=".NET 10.0.8", isa="AVX2,FMA", affinity="0x4", settings=dict(jit="default-tiered", gc="workstation:Interactive", variables={"DOTNET_TieredCompilation": "1"}))
    runner = {"runner.dll": "c" * 64}
    runner_hash = hashlib.sha256(("runner.dll\0" + "c" * 64 + "\n").encode()).hexdigest()
    origin = dt.datetime(2026, 10, 16, tzinfo=dt.timezone.utc)
    serial = 0

    def worker(role, name, mode):
        nonlocal serial
        serial += 1
        case = next(c for c in oracle["cases"] if c["id"] == name)
        stem = f"worker-{serial}"
        paths = {key: root / (stem + suffix) for key, suffix in (("process", ".json"), ("config", ".config.json"),
            ("supervision", ".supervision.json"), ("log", ".log"), ("error", ".err.log"), ("before", ".pre.json"), ("after", ".post.json"))}
        config = dict(root=str(root), fixture=str(root / "fixtures/fixtures.json"), fixture_sha256=fixture_hash,
            case=name, cpu=2, output=str(paths["process"]), dimensions=768, packed_weight_budget=67108864,
            source_sha="1" * 40, core_sha256=core, smoke=True)
        save(paths["config"], config)
        start = origin + dt.timedelta(seconds=serial * 2)
        end = start + dt.timedelta(seconds=1)
        parent = dict(id=9999, parent=0, start=1, cpu=0., name="synthetic supervisor")
        child = dict(id=10000 + serial, parent=9999, start=100 + serial, cpu=0., name="synthetic child")
        before = dict(monotonic=float(serial * 2), utc=start.isoformat(), boot="synthetic", logical_cpus=4, vanished=0, processes=[parent, child])
        after = copy.deepcopy(before)
        after["monotonic"] += 1
        after["processes"][1]["cpu"] = .5
        save(paths["before"], before); save(paths["after"], after)
        native = None if mode == "lok" else dict(path="/synthetic/runtimes/linux-x64/native/libonnxruntime.so", sha256="d" * 64, architecture="x64")
        record = dict(producer=lane.PROTOCOL, timing_contract=lane.CONTRACT, mode=mode, case=name, smoke=True,
            process_id=child["id"], started_utc=(start + dt.timedelta(milliseconds=1)).isoformat(),
            completed_utc=(end - dt.timedelta(milliseconds=1)).isoformat(), exit_code=0,
            configuration_sha256=evidence.sha256(paths["config"]), source_sha="1" * 40, core_sha256=core,
            environment=env, runner_files=runner, runner_sha256=runner_hash, dimensions=768, packed_weight_budget=67108864,
            fixture_sha256=fixture_hash, assets={"onnx/model.onnx": lane.MODEL, "onnx/model.onnx_data": lane.WEIGHTS,
                "tokenizer.json": lane.TOKENIZER, "prepared-fp32/model.onnx": "a" * 64, "prepared-fp32/model.onnx_data": lane.WEIGHTS,
                "prepared-fp32/preparation.json": "b" * 64}, native_module=native, ort_version=None if mode == "lok" else "1.29.0",
            input_sha256=lane.input_hash(root / "fixtures", case), inputs_intact=True, confinement=dict(ratio=1.), stopwatch_frequency=1000,
            measured=dict(pre_component_allowance_ratio=.1, post_component_allowance_ratio=.1, warmup_ticks=[10], warmup_wall_ticks=11,
                first_execute_ticks=12, post_execute_ticks=12, warmup_stop="fixed-smoke", blocks=[dict(index=0, block_ticks=21,
                    execute_ticks=[10, 10], allocated_bytes=100, process_cpu_ns=20000000, thread_cpu_ns=20000000, gen0=0, gen1=0, gen2=0, gc_pause_ticks=0)]))
        save(paths["process"], record)
        save(paths["supervision"], dict(supervisor_pid=9999, pid=child["id"], mode=mode, case=name, exit_code=0,
            launched_utc=start.isoformat(), exited_utc=end.isoformat(), command=["dotnet", str(root / "staged" / role / "Lokad.Onnx.EmbeddingGemma2Bench.dll"), mode, str(paths["config"])],
            accounting=processes.foreign_fraction(before, after, 9999)))
        paths["log"].write_text("synthetic; no measurements\n"); paths["error"].write_text("")
        return {key: isolated.binding(path, root) for key, path in paths.items()}

    manifest = dict(schema=5, protocol=lane.PROTOCOL, kind="smoke", core_builds=builds, producer_files=producers,
        cooldown_seconds=0, started_utc=origin.isoformat(), preflight=[worker("L0", lane.CASES[0], mode) for mode in ("lok", "ort")], visits=[])
    for index, role in enumerate(("L0", "L1")):
        engines = ("lok", "ort") if index == 0 else ("ort", "lok")
        manifest["visits"].append(dict(role=role, rep=1, workers=[worker(role, name, mode) for name in lane.CASES for mode in engines]))
    manifest["completed_utc"] = (origin + dt.timedelta(seconds=serial * 2 + 2)).isoformat()
    save(root / "evidence.json", manifest)
    return manifest


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "campaign"
        self.manifest = make_smoke(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_smoke_is_valid_but_cannot_calibrate(self):
        value = lane.load(self.root / "evidence.json")
        self.assertEqual(value["cases"], lane.CASES)
        with self.assertRaises(evidence.EvidenceError): lane.calibration(value)

    def test_missing_case_is_rejected(self):
        self.manifest["visits"][0]["workers"].pop()
        save(self.root / "evidence.json", self.manifest)
        with self.assertRaises(evidence.EvidenceError): lane.load(self.root / "evidence.json")

    def test_changed_binary_is_rejected(self):
        (self.root / "staged/L1/Lokad.Onnx.dll").write_bytes(b"changed")
        with self.assertRaises(evidence.EvidenceError): lane.load(self.root / "evidence.json")

    def test_changed_input_bytes_are_rejected(self):
        (self.root / "fixtures/gemma2-32tok/input_ids").write_bytes(b"changed")
        with self.assertRaises(evidence.EvidenceError): lane.load(self.root / "evidence.json")

    def test_missing_samples_are_rejected_even_with_updated_file_binding(self):
        entry = self.manifest["visits"][0]["workers"][0]
        path = self.root / entry["process"]["file"]
        record = isolated.read_json(path)
        record["measured"]["blocks"][0]["execute_ticks"].pop()
        save(path, record); entry["process"] = isolated.binding(path, self.root)
        save(self.root / "evidence.json", self.manifest)
        with self.assertRaises(evidence.EvidenceError): lane.load(self.root / "evidence.json")

    def test_managed_worker_cannot_declare_a_native_runtime(self):
        entry = self.manifest["visits"][0]["workers"][0]
        path = self.root / entry["process"]["file"]
        record = isolated.read_json(path)
        record["native_module"] = dict(path="/fake/libonnxruntime.so", sha256="d" * 64, architecture="x64")
        save(path, record); entry["process"] = isolated.binding(path, self.root)
        save(self.root / "evidence.json", self.manifest)
        with self.assertRaises(evidence.EvidenceError): lane.load(self.root / "evidence.json")


if __name__ == "__main__":
    unittest.main()
