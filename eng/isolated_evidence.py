"""Read-only validation of prospective isolated-e5-v1 (schema 3) evidence.

Workers remain separate processes in the evidence. Only after verifying every
binding do we assemble the logical role visits consumed by the existing scorer.
No worker timings are averaged, filtered, corrected or synthesized.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct

import campaign_evidence as base
import campaign_processes as processes

PROTOCOL = "isolated-e5-v1"
CONTRACT = "public-execute-v1; all individual calls retained; reset/disposal outside; complete-request blocks separate"
MODEL = "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"
CASES = base.CASES[:5]
TOKENS = dict(zip(CASES, (8, 30, 128, 128, 512)))
REAL = dict(zip(CASES, (8, 30, 30, 128, 512)))
EXECUTION = dict(managed="auto", threads=1, ort_provider="cpu-only", ort_optimizations="ORT_ENABLE_ALL",
                 intraop=1, interop=1, spinning=False, sequential=True)
SCRIPTS = ("run_isolated_e5.py", "isolated_evidence.py", "campaign_evidence.py", "campaign_processes.py",
           "run_common_campaign.py", "score_campaign.py", "parse_baseline.py")
require = base.require


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=base._unique_json,
                      parse_constant=base._bad_constant)


def binding(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    require(path.is_relative_to(directory), "bound file must be inside evidence directory")
    return {"file": path.relative_to(directory).as_posix(), "sha256": base.sha256(path)}


def bound(directory, value):
    require(isinstance(value, dict) and set(value) == {"file", "sha256"}, "invalid file binding")
    name = value["file"]
    require(isinstance(name, str) and name and not any(c in name for c in "\\:\0")
            and not name.startswith("/") and ".." not in name.split("/"), "invalid evidence-relative path")
    path = (directory / name).resolve()
    require(path.is_relative_to(directory.resolve()), "bound file escapes evidence directory")
    require(base.sha256(path) == base.digest(value["sha256"], "bound file hash"), "bound file digest mismatch: " + name)
    return path


def name_of(path):
    require(isinstance(path, str) and path, "path missing")
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def environment(record):
    env = record["environment"]
    require(set(env) == {"host", "cpu", "os", "architecture", "sdk", "runtime", "isa", "affinity", "settings"}, "environment fields differ")
    for key in set(env) - {"settings"}:
        require(isinstance(env[key], str) and env[key].strip(), "empty environment field: " + key)
    require(re.fullmatch(r"\.NET \d+\.\d+\.\d+", env["runtime"]), "runtime patch missing")
    require(env["architecture"] in ("x64", "arm64"), "unsupported process architecture")
    mask = int(env["affinity"], 16)
    require(mask > 0 and mask & (mask - 1) == 0, "affinity is not one CPU")
    settings = env["settings"]
    require(set(settings) == {"jit", "gc", "variables"}, "runtime settings fields differ")
    require(settings["jit"] in ("full-opts", "default-tiered") and isinstance(settings["gc"], str) and settings["gc"], "JIT/GC identity missing")
    require(isinstance(settings["variables"], dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in settings["variables"].items()), "runtime variables invalid")
    tiered = [v for k, v in settings["variables"].items() if k.lower() in ("dotnet_tieredcompilation", "complus_tieredcompilation")]
    require(tiered and all(v == ("0" if settings["jit"] == "full-opts" else "1") for v in tiered), "JIT label contradicts tiering variables")
    return env


def runner_identity(record):
    files = record["runner_files"]
    require(isinstance(files, dict) and files, "runner identities missing")
    for name, value in files.items():
        require(isinstance(name, str) and name and not any(c in name for c in "\\/:\0\n"), "invalid runner filename")
        base.digest(value, "runner file digest")
    text = "".join(name + "\0" + files[name].lower() + "\n" for name in sorted(files))
    require(hashlib.sha256(text.encode()).hexdigest() == base.digest(record["runner_sha256"], "runner digest"), "runner composite differs")


def native_identity(native, env):
    path = native["path"]
    require(isinstance(path, str) and re.match(r"^(?:/|[A-Za-z]:[\\/]|\\\\)", path), "native path must be absolute")
    require(re.fullmatch(r"onnxruntime\.dll|libonnxruntime\.so(?:\.[0-9.]+)?", name_of(path), re.IGNORECASE), "not a native ORT module")
    require(native["architecture"] == env["architecture"], "native/process architecture mismatch")
    rid = re.search(r"/runtimes/[^/]+-(x64|arm64)/native/", path.replace("\\", "/"), re.IGNORECASE)
    require(rid is None or rid[1].lower() == env["architecture"], "native RID/process architecture mismatch")
    return base.digest(native["sha256"], "native ORT digest")


def sample_ticks(value, label, count=None):
    require(isinstance(value, list) and value, label + " must be a nonempty tick array")
    require(count is None or len(value) == count, label + " count differs")
    for tick in value:
        base.integer(tick, label, 1)
    return value


def validate_measurements(record, smoke):
    measured = record["measured"]
    if record["mode"] == "oracle":
        require(measured == {}, "oracle cannot contain timed samples")
        return []
    frequency = base.integer(record["stopwatch_frequency"], "stopwatch frequency", 1)
    for field in ("pre_scaled_error", "post_scaled_error"):
        require(base.number(measured[field], field) <= 1e-4, "numerical agreement failed")
    warm = sample_ticks(measured["warmup_ticks"], "warmup")
    require((len(warm) == 1 and measured["warmup_stop"] == "fixed-smoke") if smoke else
            (9 <= len(warm) <= 1000 and measured["warmup_stop"] == "steady"), "warmup count/stop differs")
    if not smoke:
        require(sum(warm) >= frequency, "warmup below 1000ms")
        require(max(warm[-9:]) - min(warm[-9:]) <= 0.1 * statistics.median(warm[-9:]), "warmup last-nine window did not converge")
        # The last bounded call may finish after the sixty-second deadline.
        require(sum(warm[:-1]) <= 60 * frequency, "warmup continued past wall cap")
    blocks = measured["blocks"]
    require(isinstance(blocks, list) and len(blocks) == (1 if smoke else 3), "block count differs")
    ticks = []
    block_total = 0
    for index, block in enumerate(blocks):
        require(type(block["index"]) is int and block["index"] == index, "block order differs")
        raw = sample_ticks(block["execute_ticks"], "Execute/Run", 2 if smoke else 11)
        duration = base.integer(block["block_ticks"], "block ticks", 1)
        require(duration >= sum(raw), "Execute/Run ticks exceed block duration")
        for field in ("allocated_bytes", "thread_cpu_ns", "process_cpu_ns", "gen0", "gen1", "gen2", "gc_pause_ticks"):
            base.integer(block[field], field)
        block_total += duration
        ticks.extend(raw)
    total = sum(warm) + block_total + base.integer(record["load_ticks"], "load ticks", 1)
    total += base.integer(measured["first_execute_ticks"], "first Execute ticks", 1)
    total += base.integer(measured["post_execute_ticks"], "post Execute ticks", 1)
    elapsed = (base.timestamp(record["completed_utc"]) - base.timestamp(record["started_utc"])).total_seconds()
    require(total / frequency + record["confinement"]["wall_ms"] / 1000 <= elapsed + 0.001, "timings exceed process duration")
    return [tick * 1000 / frequency for tick in ticks]


def validate_process(directory, entry, smoke):
    require(set(entry) == {"process", "supervision", "config", "log", "error", "before", "after"}, "process binding set differs")
    paths = {key: bound(directory, value) for key, value in entry.items()}
    record, supervision, config = (read_json(paths[key]) for key in ("process", "supervision", "config"))
    require(record["producer"] == PROTOCOL and record["timing_contract"] == CONTRACT, "worker protocol differs")
    require(record["smoke"] is smoke and record["mode"] in ("oracle", "lok", "ort") and record["case"] in CASES, "worker scope differs")
    require(record["exit_code"] == supervision["exit_code"] == 0 and type(record["exit_code"]) is int, "worker failed")
    require(base.integer(record["process_id"], "process ID", 1) == supervision["pid"], "supervised PID differs")
    start, end = base.timestamp(record["started_utc"]), base.timestamp(record["completed_utc"])
    launch, exit = base.timestamp(supervision["launched_utc"]), base.timestamp(supervision["exited_utc"])
    require(launch <= start < end <= exit, "worker outside supervised interval")
    require(supervision["mode"] == record["mode"] and supervision["case"] == record["case"], "supervised mode/case differs")
    require(record["configuration_sha256"] == base.sha256(paths["config"]), "worker configuration digest differs")
    require(set(config) == {"root", "case", "cpu", "output", "source_sha", "core_sha256", "fixture", "fixture_sha256", "smoke"}, "configuration fields differ")
    require(config["case"] == record["case"] and config["smoke"] is smoke, "configuration case/smoke differs")
    require(config["source_sha"] == record["source_sha"] and config["core_sha256"] == record["core_sha256"], "configuration core differs")
    require(base.digest(record["source_sha"], "source commit", 40) and base.digest(record["core_sha256"], "core digest"), "core identity missing")
    require(name_of(config["output"]) == paths["process"].name, "configuration output differs")
    require((config["fixture_sha256"] is None) if record["mode"] == "oracle" else
            (config["fixture_sha256"] == record["fixture_sha256"]), "configuration oracle digest differs")
    command = supervision["command"]
    require(isinstance(command, list) and len(command) == 5 and command[0] == "dotnet" and command[2:4] == ["isolate", record["mode"]]
            and name_of(command[4]) == paths["config"].name and name_of(command[1]) == "Lokad.Onnx.Campaign.dll", "supervised command differs")
    before, after = read_json(paths["before"]), read_json(paths["after"])
    accounting = processes.foreign_fraction(before, after, base.integer(supervision["supervisor_pid"], "supervisor PID", 1))
    require(accounting == supervision["accounting"], "process accounting does not match snapshots")
    require(base.timestamp(before["utc"]) <= launch and exit <= base.timestamp(after["utc"]), "snapshots do not enclose child")
    require(smoke or accounting["foreign_cpu_fraction"] <= 0.10, "foreign CPU contamination")
    env = environment(record)
    require(type(config["cpu"]) is int and 0 <= config["cpu"] < 64 and int(env["affinity"], 16) == 1 << config["cpu"], "configuration affinity differs")
    runner_identity(record)
    require(record["execution"] == EXECUTION, "engine execution settings differ")
    for key in ("threads", "intraop", "interop"):
        base.integer(record["execution"][key], key, 1)
    require(record["execution"]["spinning"] is False and record["execution"]["sequential"] is True, "native execution flags differ")
    require(record["inputs_intact"] is True and record["external_data"] == {} and record["model_sha256"] == MODEL, "model/input contract failed")
    for key in ("tokenizer_sha256", "input_sha256", "fixture_sha256", "oracle_native_sha256"):
        base.digest(record[key], key)
    require(type(record["unmasked_tokens"]) is int and record["unmasked_tokens"] == REAL[record["case"]], "wrong unmasked token count")
    require(isinstance(record["ort_version"], str) and re.fullmatch(r"\d+\.\d+\.\d+", record["ort_version"]), "ORT version missing")
    if record["mode"] == "lok":
        require(record["native_module"] is None, "Lokad worker loaded native ORT")
    else:
        require(native_identity(record["native_module"], env) == record["oracle_native_sha256"], "native module differs from oracle")
    confinement = record["confinement"]
    wall, cpu, ratio = (base.number(confinement[field], field) for field in ("wall_ms", "cpu_ms", "ratio"))
    require(wall > 0 and cpu >= 0 and abs(cpu / wall - ratio) <= 0.02, "confinement counter arithmetic differs")
    require(smoke or (wall >= 1900 and cpu > 0 and 0.8 <= ratio <= 1.3), "confinement check failed")
    samples = validate_measurements(record, smoke)
    return dict(record, raw=samples, launch=launch, exit=exit, accounting=accounting, evidence_path=str(paths["process"]))


def validate_fixture(directory, value, oracle):
    path = bound(directory, value)
    fixture = read_json(path)
    require(base.sha256(path) == oracle["fixture_sha256"], "oracle fixture binding differs")
    require(fixture["protocol"] == PROTOCOL and fixture["case"] == oracle["case"], "oracle fixture scope differs")
    for field in ("model_sha256", "tokenizer_sha256", "input_sha256", "unmasked_tokens"):
        require(fixture[field] == oracle[field], "oracle fixture identity differs: " + field)
    require(fixture["oracle_version"] == oracle["ort_version"] and fixture["native"] == oracle["native_module"], "oracle native identity differs")
    outputs = fixture["outputs"]
    require(isinstance(outputs, list) and len(outputs) == 1, "e5 oracle output set differs")
    output = outputs[0]
    require(output["name"] == "last_hidden_state" and output["dims"] == [1, TOKENS[oracle["case"]], 384]
            and output["dtype"] == "float32" and output["file"] == "output-0.f32", "e5 oracle output metadata differs")
    payload = path.parent / output["file"]
    require(base.sha256(payload) == base.digest(output["sha256"], "oracle output digest"), "oracle output digest differs")
    raw = payload.read_bytes()
    require(len(raw) == TOKENS[oracle["case"]] * 384 * 4, "oracle output size differs")
    require(all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", raw)), "oracle output is non-finite")
    return {key: fixture[key] for key in ("model_sha256", "tokenizer_sha256", "input_sha256", "unmasked_tokens", "outputs")}


def load(path, manifest, allow_smoke=False):
    path = Path(path).resolve()
    directory = path.parent
    require(manifest["schema"] == 3 and manifest["scope"] == "e5" and manifest["protocol"] == PROTOCOL, "isolated campaign scope/protocol differs")
    smoke = manifest["kind"] == "smoke"
    require(manifest["kind"] in ("aa", "comparison") or (smoke and allow_smoke), "smoke cannot score")
    require(type(manifest["cooldown_seconds"]) is int and manifest["cooldown_seconds"] == (0 if smoke else 300), "pair cooldown contract differs")
    require(set(manifest["producer_files"]) == set(SCRIPTS), "supervisor/scorer source provenance missing")
    producer_files = {}
    for name, value in manifest["producer_files"].items():
        script = bound(directory, value)
        require(script.name == name, "producer filename differs")
        producer_files[name] = value["sha256"]
    require(set(manifest["builds"]) == {"L0", "L1"}, "build provenance missing")
    for build in manifest["builds"].values():
        require(set(build) == {"source_sha", "core_sha256", "source_archive_sha256", "sdk"}, "build provenance fields differ")
        base.digest(build["source_sha"], "build source", 40)
        base.digest(build["core_sha256"], "build core")
        base.digest(build["source_archive_sha256"], "build archive")
        require(isinstance(build["sdk"], str) and build["sdk"], "build SDK missing")
    seen_paths, seen_processes, signatures, case_ids, identities = set(), set(), [], {}, {}
    previous_exit = None

    def child(entry, expected_mode, expected_case):
        nonlocal previous_exit
        row = validate_process(directory, entry, smoke)
        require(row["mode"] == expected_mode and row["case"] == expected_case, "worker order/mode/case differs")
        require(previous_exit is None or row["launch"] >= previous_exit, "worker intervals overlap or run out of order")
        previous_exit = row["exit"]
        identity = row["process_id"], row["started_utc"]
        paths = {entry[key]["file"] for key in entry}
        require(identity not in seen_processes and len(paths) == len(entry) and not paths.intersection(seen_paths), "reused worker evidence")
        seen_processes.add(identity); seen_paths.update(paths)
        signature = {key: row[key] for key in ("environment", "runner_sha256", "ort_version", "oracle_native_sha256", "execution", "timing_contract")}
        require(not signatures or signature == signatures[0], "worker host/runner/native/settings differ")
        signatures.append(signature)
        return row

    require(len(manifest["oracles"]) == len(CASES), "five oracle processes required")
    oracles = {}
    for name, entry in zip(CASES, manifest["oracles"]):
        require(set(entry) == {"worker", "fixture"}, "oracle binding fields differ")
        row = child(entry["worker"], "oracle", name)
        oracles[name] = row
        case_ids[name] = validate_fixture(directory, entry["fixture"], row)
    count = 2 if smoke else 8
    require(isinstance(manifest["runs"], list) and len(manifest["runs"]) == count, "role visit count differs")
    visits = []
    campaign_start = None
    for index, visit in enumerate(manifest["runs"]):
        require(visit["role"] == base.ORDER[index] and type(visit["rep"]) is int and visit["rep"] == index // 2 + 1, "role visit order differs")
        require(len(visit["workers"]) == len(CASES) * 2, "ten fresh engine workers required per visit")
        engines = ("lok", "ort") if index % 2 == 0 else ("ort", "lok")
        parsed = {name: {} for name in CASES}
        start = None
        preceding_exit = previous_exit
        for worker_index, (name, mode) in enumerate((name, mode) for name in CASES for mode in engines):
            row = child(visit["workers"][worker_index], mode, name)
            if start is None:
                start = row["launch"]
                if index > 0 and index % 2 == 0:
                    require((start - preceding_exit).total_seconds() >= 300, "pair cooldown was shortened")
            oracle = oracles[name]
            for field in ("fixture_sha256", "model_sha256", "tokenizer_sha256", "input_sha256", "unmasked_tokens"):
                require(row[field] == oracle[field], "worker differs from independent oracle: " + field)
            identity = (row["source_sha"], row["core_sha256"])
            role = visit["role"]
            require(role not in identities or identities[role] == identity, "core/source changes within role")
            identities[role] = identity
            build = manifest["builds"][role]
            require(identity == (build["source_sha"], build["core_sha256"]) and row["environment"]["sdk"] == build["sdk"], "worker/build provenance differs")
            parsed[name][mode] = row["raw"]
        visits.append(dict(role=visit["role"], rep=visit["rep"], parsed={"raw": parsed}, cases_failed=[],
                           started_utc=start.isoformat(), completed_utc=previous_exit.isoformat()))
        if campaign_start is None:
            campaign_start = start
    require(all((row["source_sha"], row["core_sha256"]) == identities["L0"] for row in oracles.values()), "oracles must use baseline runner/core stage")
    if manifest["kind"] == "aa":
        require(identities["L0"] == identities["L1"], "A/A cores differ")
    signature = dict(signatures[0], schema=3, scope="e5", cases=case_ids, producer_files=producer_files)
    return dict(manifest=str(path), manifest_sha256=base.sha256(path), kind=manifest["kind"], schema=3, scope="e5", cases=CASES,
                runs=visits, signature=signature, identities=identities, start=campaign_start, end=previous_exit)


def calibration_health(campaign):
    import score_campaign as scorer
    require(campaign["kind"] == "aa" and campaign["schema"] == 3, "isolated calibration requires schema-3 A/A")
    problems = scorer.campaign_stability(campaign, "A/A")
    noise = {}
    for name in CASES:
        pairs = scorer.medians(campaign, name)
        controls = {key: scorer.spread([pair[key + str(i)] for pair in pairs for i in (0, 1)]) for key in ("L", "O", "R")}
        noise[name] = controls
        if max(controls.values()) > scorer.NOISE_CAP:
            problems.append("A/A %s variation %.4f exceeds %.2f" % (name, max(controls.values()), scorer.NOISE_CAP))
    return dict(qualified=not problems, problems=problems, spreads=noise)
