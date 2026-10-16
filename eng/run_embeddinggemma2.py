"""Isolated owned-vector campaigns using the existing staging, accounting, and scorer policy."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import sys
import time

import campaign_evidence as evidence
import campaign_processes as processes
import isolated_evidence as isolated
import run_common_campaign as common
import score_campaign

PROTOCOL = "embeddinggemma2-owned-vector-v1"
CONTRACT = "reused-context-or-optimized-original-ORT; owned-final-vector; pool/truncate/normalize/copy inside; reset/disposal outside; 33 calls retained"
CASES = ("gemma2-32tok", "gemma2-128tok", "gemma2-512tok", "gemma2-32pad128")
WIDTHS = dict(zip(CASES, (32, 128, 512, 128)))
REAL = dict(zip(CASES, (32, 128, 512, 32)))
SCRIPTS = ("run_embeddinggemma2.py", *isolated.SCRIPTS)
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
        require(sum(warm) / frequency >= 1 and len(warm) >= 9 and len(warm) <= 1000, "insufficient warmup")
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


def identity_view(record, producers):
    current = {key: record[key] for key in ("environment", "runner_sha256", "runner_files", "dimensions", "packed_weight_budget",
               "assets", "fixture_sha256", "timing_contract", "stopwatch_frequency")}
    current["producer_files"] = producers
    return current


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


def load(path):
    path = Path(path).resolve()
    root = isolated.read_json(path)
    require(root["schema"] == 5 and root["protocol"] == PROTOCOL and root["kind"] in ("smoke", "aa", "comparison"), "unknown campaign")
    directory, smoke = path.parent, root["kind"] == "smoke"
    require([item["name"] for item in root["producer_files"]] == list(SCRIPTS), "producer source set differs")
    producers = {}
    for item in root["producer_files"]:
        producers[item["name"]] = evidence.sha256(isolated.bound(directory, item["binding"]))
    manifest = fixture(directory / "fixtures")
    visits, signature, identities, pids, native = [], None, {}, set(), []
    previous_end = None
    require(len(root["preflight"]) == 2 and set(root["core_builds"]) == {"L0", "L1"}, "missing preflight/build evidence")
    builds = {}
    for role, item in root["core_builds"].items():
        build = isolated.read_json(isolated.bound(directory, item["build"]))
        require(evidence.sha256(isolated.bound(directory, item["source_archive"])) == build["source_archive_sha256"], "source archive differs")
        require(evidence.sha256(directory / "staged" / role / "Lokad.Onnx.dll") == build["core_sha256"], "staged binary differs from build")
        builds[role] = build
    for entry, mode in zip(root["preflight"], ("lok", "ort")):
        record, _, supervision = worker(directory, entry, True, manifest)
        require((record["case"], record["mode"]) == (CASES[0], mode), "preflight case/engine differs")
        require((record["source_sha"], record["core_sha256"]) == (builds["L0"]["source_sha"], builds["L0"]["core_sha256"]), "preflight baseline differs")
        current = identity_view(record, producers)
        require(signature is None or signature == current, "preflight identities differ")
        signature = current
        pids.add(record["process_id"])
        previous_end = evidence.timestamp(supervision["exited_utc"])
        if mode == "ort": native.append(record["native_module"]["sha256"])
    order = evidence.ORDER[:2] if smoke else evidence.ORDER
    require(len(root["visits"]) == len(order), "campaign visit coverage differs")
    for visit_index, visit in enumerate(root["visits"]):
        role, rep = order[visit_index], visit_index // 2 + 1
        require((visit["role"], visit["rep"]) == (role, rep), "balanced role order differs")
        engines = ("lok", "ort") if visit_index % 2 == 0 else ("ort", "lok")
        expected = [(name, mode) for name in CASES for mode in engines]
        require(len(visit["workers"]) == len(expected), "visit worker coverage differs")
        rows = {name: {} for name in CASES}
        for worker_index, (entry, (name, mode)) in enumerate(zip(visit["workers"], expected)):
            record, raw, supervision = worker(directory, entry, smoke, manifest)
            require((record["case"], record["mode"]) == (name, mode), "case/engine order differs")
            require(record["process_id"] not in pids, "worker PID reused")
            pids.add(record["process_id"])
            start, end = (evidence.timestamp(supervision[k]) for k in ("launched_utc", "exited_utc"))
            require(previous_end is None or previous_end <= start, "overlapping workers")
            if not smoke and visit_index and visit_index % 2 == 0 and worker_index == 0:
                require((start - previous_end).total_seconds() >= 300, "observed pair cooldown was too short")
            previous_end = end
            current = identity_view(record, producers)
            require(signature is None or signature == current, "worker host/runner/configuration differs")
            signature = current
            identity = (record["source_sha"], record["core_sha256"])
            require(identity == (builds[role]["source_sha"], builds[role]["core_sha256"]) and
                    record["environment"]["sdk"] == builds[role]["sdk"], "worker differs from archived source/build SDK")
            require(role not in identities or identities[role] == identity, "role binary/source changed")
            identities[role] = identity
            rows[name][mode] = raw
            if mode == "ort": native.append(record["native_module"]["sha256"])
        visits.append({"role": role, "rep": rep, "parsed": {"raw": rows}})
    require(len(set(native)) == 1, "native ORT binary changed")
    signature["native_sha256"] = native[0]
    signature["cases"] = {c["id"]: c for c in manifest["cases"]}
    require(root["kind"] != "aa" or identities["L0"] == identities["L1"], "A/A cores differ")
    require(smoke or root["cooldown_seconds"] >= 300, "scored pair cooldown differs")
    require(evidence.timestamp(root["started_utc"]) <= evidence.timestamp(root["completed_utc"]), "invalid campaign interval")
    first = isolated.read_json(isolated.bound(directory, root["preflight"][0]["supervision"]))
    require(evidence.timestamp(root["started_utc"]) <= evidence.timestamp(first["launched_utc"]) and
            previous_end <= evidence.timestamp(root["completed_utc"]), "workers outside campaign interval")
    return {"kind": root["kind"], "schema": 5, "scope": "embeddinggemma2", "cases": CASES,
        "runs": visits, "signature": signature, "identities": identities,
        "start": evidence.timestamp(root["started_utc"]), "end": evidence.timestamp(root["completed_utc"]),
        "manifest_sha256": evidence.sha256(path)}


def calibration(aa):
    require(aa["kind"] == "aa", "calibration requires A/A evidence")
    issues = score_campaign.campaign_stability(aa, "A/A")
    for name in CASES:
        pairs = score_campaign.medians(aa, name)
        variation = max(score_campaign.spread([row[key + str(i)] for row in pairs for i in (0, 1)]) for key in ("L", "O", "R"))
        if variation > .03:
            issues.append(f"{name}: unchanged-run variation {variation:.4f} exceeds 3%")
    require(not issues, "A/A is inconclusive: " + "; ".join(issues))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=common.ROOT)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--kind", choices=("smoke", "aa", "comparison"), required=True)
    parser.add_argument("--aa", type=Path)
    parser.add_argument("--dimensions", type=int, choices=(128, 256, 512, 768), default=768)
    parser.add_argument("--packed-weight-budget", type=int, default=576 * 1024 * 1024)
    parser.add_argument("--jit", choices=("default-tiered", "full-opts"), default="default-tiered")
    args = parser.parse_args()
    require(0 <= args.cpu < 64 and args.packed_weight_budget >= 0, "invalid CPU/budget")
    require(args.kind != "comparison" or args.aa is not None, "comparison requires preceding A/A")
    aa = load(args.aa) if args.aa else None
    if aa is not None: calibration(aa)
    source_fixture = args.fixtures.resolve()
    fixture(source_fixture.parent)
    output = args.output.resolve()
    require(not output.is_relative_to(source_fixture.parent), "campaign output must be outside the fixture directory")
    builds = common.stage(args.prepared.resolve(), output, "aa" if args.kind == "aa" else "comparison", "Lokad.Onnx.EmbeddingGemma2Bench.dll")
    shutil.copytree(source_fixture.parent, output / "fixtures")
    producer_files = []
    for script in SCRIPTS:
        destination = output / "producers" / script
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).parent / script, destination)
        producer_files.append({"name": script, "binding": isolated.binding(destination, output)})
    env = dict(os.environ)
    for key in ("DOTNET_TieredCompilation", "COMPlus_TieredCompilation"):
        env[key] = "0" if args.jit == "full-opts" else "1"
    smoke = args.kind == "smoke"
    core_builds = {}
    for role in ("L0", "L1"):
        source_role = "L0" if args.kind == "aa" else role
        target = output / "builds" / role
        target.mkdir(parents=True)
        shutil.copyfile(args.prepared.resolve() / source_role / "build.json", target / "build.json")
        shutil.copyfile(args.prepared.resolve() / source_role / "source.zip", target / "source.zip")
        core_builds[role] = {"build": isolated.binding(target / "build.json", output), "source_archive": isolated.binding(target / "source.zip", output)}
    if aa is not None:
        require(tuple(aa["identities"]["L0"]) == (builds["L0"]["source_sha"], builds["L0"]["core_sha256"]), "A/A baseline differs before launch")
        require(aa["signature"]["dimensions"] == args.dimensions and aa["signature"]["packed_weight_budget"] == args.packed_weight_budget and
                aa["signature"]["fixture_sha256"] == evidence.sha256(output / "fixtures/fixtures.json"), "A/A settings/fixture differ before launch")
    manifest = dict(schema=5, protocol=PROTOCOL, kind=args.kind, visits=[], preflight=[], core_builds=core_builds, producer_files=producer_files,
        cooldown_seconds=0 if smoke else 300, started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    producer_hashes = {p["name"]: p["binding"]["sha256"] for p in producer_files}
    oracle = fixture(output / "fixtures")
    for index, mode in enumerate(("lok", "ort")):
        entry = child(builds["L0"], args.root.resolve(), output, CASES[0], mode, args.cpu, args.dimensions,
                      args.packed_weight_budget, True, env, index, prefix="probe")
        manifest["preflight"].append(entry)
        record, _, _ = worker(output, entry, True, oracle)
        if aa is not None:
            expected = {k: v for k, v in aa["signature"].items() if k not in ("cases", "native_sha256")}
            require(identity_view(record, producer_hashes) == expected, "A/A host/runner identity differs before measurements")
            if mode == "ort": require(record["native_module"]["sha256"] == aa["signature"]["native_sha256"], "A/A native binary differs before measurements")
    count = 0
    for visit_index, role in enumerate(evidence.ORDER[:2] if smoke else evidence.ORDER):
        if visit_index and visit_index % 2 == 0:
            for remaining in range(300, 0, -30):
                print(f"Pair cooldown: {remaining}s", flush=True); time.sleep(min(30, remaining))
        visit = dict(role=role, rep=visit_index // 2 + 1, workers=[])
        for name in CASES:
            for mode in (("lok", "ort") if visit_index % 2 == 0 else ("ort", "lok")):
                count += 1
                visit["workers"].append(child(builds[role], args.root.resolve(), output, name, mode,
                    args.cpu, args.dimensions, args.packed_weight_budget, smoke, env, count))
        manifest["visits"].append(visit)
    manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    common.save_json(output / "evidence.json", manifest)
    campaign = load(output / "evidence.json")
    if args.kind == "comparison":
        result, code = score_campaign.score(campaign, aa, primary=CASES)
        result["policy"] = "embeddinggemma2-owned-vector-v1"
        result["embeddinggemma2_improvement"] = result.pop("e5_improvement")
        common.save_json(output / "verdict.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return code
    if args.kind == "aa": calibration(campaign)
    print("PASS " + args.kind + (" (unscored; no performance claim)" if smoke else " calibration"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
