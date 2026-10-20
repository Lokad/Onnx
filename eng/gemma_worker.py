"""Gemma fixture, launch and numerical validation shared by routine and strict benchmarks."""
import datetime as dt
import hashlib
import os
from pathlib import Path
import statistics
import struct

import benchmark_worker as common
import campaign_evidence as evidence
import campaign_processes as processes
import isolated_evidence as isolated

PROTOCOL = "embeddinggemma2-owned-vector-v1"
CONTRACT = "reused-context-or-optimized-original-ORT; owned-final-vector; pool/truncate/normalize/copy inside; reset/disposal outside; 33 calls retained"
CASES = ("gemma2-32tok", "gemma2-128tok", "gemma2-512tok", "gemma2-32pad128")
WIDTHS = dict(zip(CASES, (32, 128, 512, 128)))
REAL = dict(zip(CASES, (32, 128, 512, 32)))
MODEL = "bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3"
WEIGHTS = "9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2"
TOKENIZER = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"
require = evidence.require


def fixture(directory):
    manifest = isolated.read_json(directory / "fixtures.json")
    require(manifest["protocol"] == PROTOCOL and manifest["schemaVersion"] == 1, "unknown fixture protocol")
    require((manifest["modelSha256"], manifest["weightsSha256"], manifest["tokenizerSha256"]) == (MODEL, WEIGHTS, TOKENIZER), "wrong FP32 source assets")
    require([c["id"] for c in manifest["cases"]] == list(CASES), "fixture case set/order differs")
    for case in manifest["cases"]:
        name = case["id"]
        require((case["sequenceLength"], case["unmaskedTokens"]) == (WIDTHS[name], REAL[name]), "fixture token counts differ")
        for key in ("input_ids", "attention_mask", *[f"embedding_{d}" for d in (128, 256, 512, 768)]):
            item = case[key]
            require(Path(item["file"]).name == item["file"], "fixture filename must be a basename")
            path = directory / name / item["file"]
            require(evidence.sha256(path) == item["sha256"], "changed fixture array")
            count = WIDTHS[name] if key in ("input_ids", "attention_mask") else int(key.split("_")[-1])
            dtype = "int64" if key in ("input_ids", "attention_mask") else "float32"
            require(item["dtype"] == dtype and item["shape"] == [1, count], "fixture array contract differs")
            require(path.stat().st_size == count * (8 if dtype == "int64" else 4), "fixture byte length differs")
    return manifest

def input_hash(directory, case):
    payload = bytearray(b"LOKAD-CAMPAIGN-INPUTS-1\0" + struct.pack("<i", 2))
    for name in ("attention_mask", "input_ids"):
        data = (directory / case["id"] / case[name]["file"]).read_bytes()
        payload += struct.pack("<i", len(name)) + name.encode() + struct.pack("<iiiiq", 7, 2, 1, case["sequenceLength"], case["sequenceLength"]) + data
    return hashlib.sha256(payload).hexdigest()

def measured(record, smoke):
    value, frequency = record["measured"], evidence.integer(record["stopwatch_frequency"], "frequency", 1)
    for key in ("pre_component_allowance_ratio", "post_component_allowance_ratio"):
        require(evidence.number(value[key], key) <= 1, "numerical agreement failed")
    warm = isolated.sample_ticks(value["warmup_ticks"], "warmup")
    warm_wall = evidence.integer(value["warmup_wall_ticks"], "warmup wall ticks", 1)
    require(warm_wall >= sum(warm), "warmup wall does not cover calls")
    evidence.integer(value["first_execute_ticks"], "first execute ticks", 1)
    evidence.integer(value["post_execute_ticks"], "post execute ticks", 1)
    require(value["warmup_stop"] == ("fixed-smoke" if smoke else "steady"), "warmup stop differs")
    if not smoke:
        require(sum(warm) / frequency >= 10 and len(warm) >= 32 and len(warm) <= 1000, "insufficient warmup")
        require(warm_wall <= 60 * frequency, "warmup exceeded wall cap")
        require((max(warm[-9:]) - min(warm[-9:])) / statistics.median(warm[-9:]) <= .10, "warmup not steady")
    require(len(value["blocks"]) == (1 if smoke else 3), "block count differs")
    raw = []
    for index, block in enumerate(value["blocks"]):
        require(block["index"] == index, "block order differs")
        ticks = isolated.sample_ticks(block["execute_ticks"], "calls", 2 if smoke else 11)
        require(evidence.integer(block["block_ticks"], "block ticks", 1) >= sum(ticks), "block does not cover calls")
        for key in ("allocated_bytes", "gen0", "gen1", "gen2", "gc_pause_ticks"):
            evidence.integer(block[key], key)
        for key in ("process_cpu_ns", "thread_cpu_ns"):
            evidence.integer(block[key], key)
        raw.extend(t * 1000 / frequency for t in ticks)
    return raw

def worker(directory, entry, smoke, manifest):
    paths = {name: isolated.bound(directory, binding) for name, binding in entry.items() if name != "role" and name != "rep"}
    require(set(paths) == {"process", "config", "supervision", "log", "error", "before", "after"}, "missing worker files")
    record, config, supervision = (isolated.read_json(paths[k]) for k in ("process", "config", "supervision"))
    require(record["producer"] == PROTOCOL and record["timing_contract"] == CONTRACT and record["smoke"] is smoke, "worker protocol differs")
    require(record["configuration_sha256"] == evidence.sha256(paths["config"]), "worker configuration differs")
    require(record["process_id"] == supervision["pid"] and supervision["exit_code"] == record["exit_code"] == 0, "worker process/exit differs")
    require(evidence.timestamp(supervision["launched_utc"]) <= evidence.timestamp(record["started_utc"]) <=
            evidence.timestamp(record["completed_utc"]) <= evidence.timestamp(supervision["exited_utc"]), "unsupervised timestamps")
    require(record["mode"] == supervision["mode"] and record["case"] == supervision["case"], "supervisor workload differs")
    command = supervision["command"]
    require(len(command) == 4 and command[0] == "dotnet" and command[2] == record["mode"] and isolated.name_of(command[3]) == paths["config"].name, "supervised command differs")
    parts = command[1].replace("\\", "/").split("/")
    role = parts[-2]
    require(role in ("L0", "L1") and parts[-3] == "staged" and parts[-1] == "Lokad.Onnx.EmbeddingGemma2Bench.dll", "supervised runner differs")
    require(evidence.sha256(directory / "staged" / role / "Lokad.Onnx.dll") == record["core_sha256"], "staged core differs")
    for key in ("source_sha", "core_sha256", "dimensions", "packed_weight_budget"):
        require(record[key] == config[key], "configuration/worker differs: " + key)
    require(record["case"] == config["case"] and record["fixture_sha256"] == config["fixture_sha256"] == evidence.sha256(directory / "fixtures/fixtures.json"), "fixture identity differs")
    env = isolated.environment(record)
    require(int(env["affinity"], 16) == 1 << config["cpu"], "selected CPU differs")
    isolated.runner_identity(record)
    assets = record["assets"]
    require(assets == {"onnx/model.onnx": MODEL, "onnx/model.onnx_data": WEIGHTS, "tokenizer.json": TOKENIZER,
        "prepared-fp32/model.onnx": manifest["preparedGraphSha256"], "prepared-fp32/model.onnx_data": WEIGHTS,
        "prepared-fp32/preparation.json": manifest["preparationSha256"]}, "worker model assets differ")
    case = next(c for c in manifest["cases"] if c["id"] == record["case"])
    require(record["input_sha256"] == input_hash(directory / "fixtures", case) and record["inputs_intact"] is True, "worker inputs differ")
    require(record["dimensions"] in (128, 256, 512, 768) and record["packed_weight_budget"] >= 0, "invalid embedding configuration")
    if record["mode"] == "lok":
        require(record["native_module"] is None and record["ort_version"] is None, "managed worker loaded ORT")
    else:
        require(record["mode"] == "ort" and record["ort_version"] == "1.29.0", "native engine differs")
        isolated.native_identity(record["native_module"], env)
    if not smoke:
        require(.8 <= record["confinement"]["ratio"] <= 1.3, "CPU confinement failed")
    before, after = (isolated.read_json(paths[k]) for k in ("before", "after"))
    accounting = processes.foreign_fraction(before, after, supervision["supervisor_pid"])
    require(accounting == supervision["accounting"], "process accounting differs")
    # Same provisional guest-noise cap as the existing isolated producer.
    require(smoke or accounting["foreign_cpu_fraction"] <= .01, "foreign guest CPU exceeds 1%")
    return record, measured(record, smoke), supervision

def child(build, root, output, name, mode, cpu, dimensions, budget, smoke, env, index, prefix="worker"):
    stem = f"{prefix}-{index:03d}-{name}-{mode}"
    paths = {key: output / (stem + suffix) for key, suffix in (("process", ".json"), ("config", ".config.json"),
        ("supervision", ".supervision.json"), ("log", ".log"), ("error", ".err.log"), ("before", ".pre.json"), ("after", ".post.json"))}
    config = dict(root=str(root), fixture=str(output / "fixtures/fixtures.json"), fixture_sha256=evidence.sha256(output / "fixtures/fixtures.json"),
        case=name, dimensions=dimensions, cpu=cpu, output=str(paths["process"]), source_sha=build["source_sha"],
        core_sha256=build["core_sha256"], packed_weight_budget=budget, smoke=smoke)
    common.save_json(paths["config"], config)
    before = processes.snapshot(); common.save_json(paths["before"], before)
    command = ["dotnet", str(build["runner"]), mode, str(paths["config"])]
    launched = dt.datetime.now(dt.timezone.utc)
    print("Starting " + stem, flush=True)
    with paths["log"].open("x", encoding="utf-8") as stdout, paths["error"].open("x", encoding="utf-8") as stderr:
        process = common.spawn_child(command, cpu, cwd=root, stdout=stdout, stderr=stderr, env=env)
        code = common.wait_child(process, stem, timeout=600)
    exited = dt.datetime.now(dt.timezone.utc)
    after = processes.snapshot(); common.save_json(paths["after"], after)
    common.save_json(paths["supervision"], dict(supervisor_pid=os.getpid(), pid=process.pid, mode=mode, case=name,
        command=command, launched_utc=launched.isoformat(), exited_utc=exited.isoformat(), exit_code=code,
        accounting=processes.foreign_fraction(before, after, os.getpid())))
    require(code == 0, "worker failed: " + str(paths["error"]))
    return {key: isolated.binding(path, output) for key, path in paths.items()}
