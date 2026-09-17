"""Strict, read-only evidence validation for the AMD E5 campaign scorer.

See campaign-evidence.md for the producer contract. No missing observation is
inferred from an old log or from the scorer's own machine.
"""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re

import parse_baseline as base

PRIMARY = ("e5-8tok", "e5-30tok", "e5-128tok", "e5-30pad128")
CASES = ("e5-8tok", "e5-30tok", "e5-30pad128", "e5-128tok", "e5-512tok",
         "dinov3-224", "resnet50-224", "gpt2-1tok", "gpt2-4tok", "gpt2-32tok",
         "gpt2-128tok", "gpt2-dec-p1", "gpt2-dec-p32", "gpt2-dec-p128", "gpt2-dec-p512")
ORDER = ("L0", "L1", "L1", "L0", "L1", "L0", "L0", "L1")
MIN_SAMPLES = 33
MIN_WARMUP_MS = 1000


class EvidenceError(ValueError):
    """Missing, inconsistent or malformed campaign evidence."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def pairs(line):
    """Preserve multiword values, notably runtime=.NET 10.0.8 and CPU names."""
    matches = list(re.finditer(r"(?<!\S)([A-Za-z][\w-]*)=", line))
    result = {}
    for i, match in enumerate(matches):
        key = match[1]
        require(key not in result, "duplicate field: " + key)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        result[key] = line[match.end():end].strip()
    return result


def number(value, label, minimum=0):
    require(type(value) in (int, float), label + " must be a number")
    require(math.isfinite(value) and value >= minimum, label + " is non-finite or below minimum")
    return value


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + " must be an integer >= " + str(minimum))
    return value


def digest(value, label, size=64):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{%d}" % size, value), label + " must be a full hex digest")
    return value.lower()


def timestamp(value):
    require(isinstance(value, str), "timestamp must be an ISO-8601 string")
    stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.utcoffset() == dt.timedelta(0), "timestamp must include UTC offset")
    return stamp


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_json(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def check_process_evidence(run, directory):
    """Bind the common producer's observations in addition to the stdout log."""
    if "producer" not in run:
        return  # Offline schema fixtures/other trusted producers remain supported.
    require(run["producer"] == "common-runner-v1", "unknown process evidence producer")
    path = directory / run["process_evidence"]
    require(sha256(path) == digest(run["process_evidence_sha256"], "process_evidence_sha256"), "process evidence digest mismatch")
    observed = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json, parse_constant=_bad_constant)
    for key in ("producer", "process_id", "started_utc", "exit_code", "source_sha", "core_sha256",
                "runner_sha256", "runner_files", "ort_native", "environment", "cases"):
        require(observed[key] == run[key], "process evidence/manifest mismatch: " + key)
    require(timestamp(observed["completed_utc"]) <= timestamp(run["completed_utc"]), "process evidence completion is after observed exit")
    files = run["runner_files"]
    require(isinstance(files, dict) and files, "runner file identities missing")
    for name, value in files.items():
        require(isinstance(name, str) and name and not any(c in name for c in "\\/:\0\n"), "invalid runner file name")
        digest(value, "runner file hash")
    encoded = "".join(name + "\0" + files[name].lower() + "\n" for name in sorted(files)).encode("utf-8")
    require(hashlib.sha256(encoded).hexdigest() == run["runner_sha256"].lower(), "runner composite digest mismatch")


def _bad_constant(value):
    raise EvidenceError("non-finite JSON number: " + value)


def _array(value, label):
    require(isinstance(value, str) and value.startswith("[") and value.endswith("]"), label + " is not an array")
    values = [float(v) for v in value[1:-1].split(",")]
    require(all(math.isfinite(v) and v > 0 for v in values), label + " requires finite positive timings")
    return values


def _put(mapping, key, value, label):
    require(key not in mapping, "duplicate " + label + ": " + key)
    mapping[key] = value


def parse_log(path):
    lines = base.read_log_text(path).splitlines()
    hosts = [line for line in lines if line.startswith("host=")]
    confs = [line for line in lines if line.startswith("confinement wallMs=")]
    require(len(hosts) == 1 and len(confs) == 1, "need exactly one host and confinement record: " + str(path))
    host = pairs(hosts[0])
    for key in ("host", "cpu", "procs", "affinity", "vector", "isa", "fma", "runtime", "lokad", "ort",
                "ort-provider", "ort-optimizations", "intraop", "interop", "mode", "threads", "rows",
                "iters", "warmup", "warmupMin", "warmupMax"):
        require(bool(host.get(key)), "missing host field: " + key)
    mask = int(host["affinity"].split()[0], 16)
    require(mask > 0 and mask & (mask - 1) == 0, "affinity must contain exactly one logical CPU")
    for key, expected in {"threads": "1", "intraop": "1", "interop": "1", "mode": "auto",
                          "rows": "canonical", "ort-provider": "cpu-only", "ort-optimizations": "ORT_ENABLE_ALL"}.items():
        require(host[key].split()[0] == expected, "unsupported execution setting: " + key)
    require(re.search(r"\bnospin\b", hosts[0]) and re.search(r"\bseq\b", hosts[0]), "ORT must be sequential with spinning disabled")
    require(re.fullmatch(r"\.NET \d+\.\d+\.\d+", host["runtime"]), "actual runtime patch is required")
    iterations = int(host["iters"])
    require(iterations >= MIN_SAMPLES, "need >=33 samples per case")
    conf = pairs(confs[0])
    wall = float(conf["wallMs"])
    cpu = float(conf["cpuMs"])
    ratio = float(conf["ratio"].split()[0])
    require(all(math.isfinite(v) for v in (wall, cpu, ratio)) and wall >= 1900 and cpu > 0,
            "invalid confinement counters")
    require(0.8 <= ratio <= 1.3 and abs(cpu / wall - ratio) <= 0.02, "confinement ratio failed")
    records = {key: {} for key in ("case-status", "casedef", "warmup", "case")}
    raw, summaries = {}, {}
    for i, line in enumerate(lines):
        for prefix, mapping in records.items():
            if line.startswith(prefix + " "):
                rest = line[len(prefix) + 1:]
                name, separator, value = rest.partition("=" if prefix == "case-status" else " ")
                require(bool(separator), "malformed " + prefix)
                _put(mapping, name, value, prefix)
        if line.startswith("raw lok="):
            require(i > 0 and " [" in lines[i - 1], "raw samples lack their case summary")
            name = lines[i - 1].split(" [", 1)[0]
            fields = pairs(line)
            # Each engine is preceded by 'raw'; remove that separator from its value.
            fields = {key: value.removesuffix(" raw") for key, value in fields.items()}
            require(set(fields) == {"lok", "ctx", "ort"}, "missing/extra raw engine")
            _put(raw, name, {key: _array(fields[key], name + " " + key) for key in fields}, "raw series")
            summaries[name] = lines[i - 1]
    for label, mapping in list(records.items()) + [("raw", raw)]:
        require(set(mapping) == set(CASES), label + " case set mismatch; missing=" + str(sorted(set(CASES) - set(mapping)))
                + " extra=" + str(sorted(set(mapping) - set(CASES))))
    require(tuple(raw) == CASES, "case order differs from canonical workload")
    defs, warmups = {}, {}
    for name in CASES:
        require(records["case-status"][name] == "ok", "case not ok: " + name)
        definition = pairs(records["casedef"][name])
        require(set(definition) == {"model", "bytes", "sha12", "inputs", "outputs", "iters", "warmup", "tol"}, "incomplete casedef: " + name)
        digest(definition["sha12"], name + " model prefix", 12)
        require(int(definition["bytes"]) > 0 and int(definition["iters"]) == iterations, "casedef size/iterations mismatch: " + name)
        require(definition["warmup"] == host["warmup"], "casedef warmup mismatch: " + name)
        require(definition["model"] and definition["inputs"] and definition["outputs"], "empty model or shapes: " + name)
        if name.startswith("e5-"):
            sequence = {"e5-8tok": 8, "e5-30tok": 30, "e5-30pad128": 128, "e5-128tok": 128, "e5-512tok": 512}[name]
            shapes = definition["inputs"].split(",")
            expected = {key + ":1x" + str(sequence) for key in ("input_ids", "attention_mask", "token_type_ids")}
            require(len(shapes) == 3 and set(shapes) == expected, "E5 input shape does not match case: " + name)
            require(definition["outputs"] == "last_hidden_state:1x%dx384" % sequence, "E5 output shape does not match case: " + name)
        tolerance = float(definition["tol"])
        require(math.isfinite(tolerance) and 0 < tolerance <= 1e-4, "invalid agreement tolerance: " + name)
        require("inputsIntact=yes" in summaries[name], "input integrity not confirmed: " + name)
        for key in ("maxScaled", "postScaled"):
            found = re.search(r"\b" + key + r"=(\S+)", summaries[name])
            require(found is not None, "missing agreement result: " + name + " " + key)
            value = float(found[1])
            require(math.isfinite(value) and 0 <= value <= tolerance, "agreement failed: " + name + " " + key)
        for engine, values in raw[name].items():
            require(len(values) == iterations, "sample count differs from iters: " + name + " " + engine)
        warm = pairs(records["warmup"][name])
        used = int(warm["used"])
        wmin, wmax = int(host["warmupMin"]), int(host["warmupMax"])
        if wmin == wmax == -1:
            require(warm["stop"] == "fixed" and used == int(host["warmup"]), "fixed warmup mismatch: " + name)
        else:
            require(0 <= wmin <= used <= wmax and warm["stop"] == "steady", "adaptive warmup did not converge: " + name)
        require(used >= 3, "too few warmup iterations: " + name)
        for engine in ("lok", "ort"):
            values = _array(warm[engine], name + " warmup " + engine)
            require(len(values) == used, "warmup count mismatch: " + name)
            require(sum(values) >= MIN_WARMUP_MS, "warmup duration below 1000 ms per engine: " + name)
        defs[name], warmups[name] = definition, warm
    return {"host": host, "definitions": defs, "warmups": warmups, "raw": raw}


def _load_campaign(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_json, parse_constant=_bad_constant)
    require(type(manifest["schema"]) is int and manifest["schema"] == 1, "unsupported evidence schema")
    require(manifest["kind"] in ("aa", "comparison"), "kind must be aa or comparison")
    runs = manifest["runs"]
    require(isinstance(runs, list) and len(runs) == 8, "need four paired repetitions (eight processes)")
    signatures, identities, seen_logs, seen_processes = [], {}, set(), set()
    previous_end = None
    for i, run in enumerate(runs):
        label = "run %d" % (i + 1)
        require(run["role"] == ORDER[i] and type(run["rep"]) is int and run["rep"] == i // 2 + 1,
                label + " must follow L0,L1 / L1,L0 / L1,L0 / L0,L1 order")
        start, end = timestamp(run["started_utc"]), timestamp(run["completed_utc"])
        require(end > start and (previous_end is None or start >= previous_end), label + " overlaps or has invalid chronology")
        previous_end = end
        process = (integer(run["process_id"], "process_id", 1), start)
        require(process not in seen_processes, "reused process identity")
        seen_processes.add(process)
        log = (path.parent / run["log"]).resolve()
        require(log not in seen_logs, "reused log path: " + str(log))
        seen_logs.add(log)
        require(digest(run["log_sha256"], "log_sha256") == sha256(log), "log digest mismatch: " + str(log))
        check_process_evidence(run, path.parent)
        require(type(run["exit_code"]) is int and run["exit_code"] == 0, label + " failed")
        require(run["accounting"]["valid"] is True, label + " has invalid process accounting")
        foreign = number(run["accounting"]["foreign_cpu_fraction"], "foreign_cpu_fraction")
        require(foreign <= 0.10, label + " has foreign CPU contamination")
        parsed = parse_log(log)
        # These loops are sequential. Their sum is a lower bound on elapsed
        # process time (load, validation and request diagnostics add more).
        measured = [value for series in parsed["raw"].values() for values in series.values() for value in values]
        for name, warm in parsed["warmups"].items():
            for engine in ("lok", "ort"):
                measured.extend(_array(warm[engine], name + " warmup " + engine))
        total_ms = sum(measured)
        require(math.isfinite(total_ms) and total_ms <= (end - start).total_seconds() * 1000 + len(measured) * 0.01,
                label + " timings exceed process duration")
        run["parsed"] = parsed
        run["log_path"] = str(log)
        identity = (digest(run["source_sha"], "source_sha", 40), digest(run["core_sha256"], "core_sha256"))
        role = run["role"]
        require(role not in identities or identities[role] == identity, "core/source changes within " + role)
        identities[role] = identity
        native = run["ort_native"]
        require(isinstance(native["path"], str) and native["path"], "loaded ORT module path missing")
        require(re.match(r"^(?:/|[A-Za-z]:[\\/]|\\\\)", native["path"]), "loaded ORT module path must be absolute")
        module = native["path"].replace("\\", "/").rsplit("/", 1)[-1]
        require(re.fullmatch(r"onnxruntime\.dll|libonnxruntime\.so(?:\.[0-9.]+)?", module, re.IGNORECASE), "ORT path is not a loaded native runtime")
        environment = run["environment"]
        require(set(environment) == {"host", "cpu", "os", "architecture", "sdk", "runtime", "isa", "affinity", "settings"}, "incomplete environment")
        for key in set(environment) - {"settings"}:
            require(isinstance(environment[key], str) and environment[key].strip(), "empty environment field: " + key)
        require(native["architecture"] == environment["architecture"], "native/process architecture mismatch")
        require(environment["architecture"] in ("x64", "arm64"), "unsupported process architecture")
        rid = re.search(r"/runtimes/[^/]+-(x64|arm64)/native/", native["path"].replace("\\", "/"), re.IGNORECASE)
        require(rid is None or rid[1].lower() == environment["architecture"], "ORT RID/process architecture mismatch")
        settings = environment["settings"]
        require(isinstance(settings, dict) and settings.get("jit") in ("default-tiered", "full-opts"), "JIT regime missing")
        require(isinstance(settings.get("variables"), dict) and isinstance(settings.get("gc"), str) and settings["gc"], "runtime/GC settings missing")
        for key, value in settings["variables"].items():
            require(isinstance(key, str) and isinstance(value, str), "runtime variables must be strings")
        tiered = [settings["variables"][key] for key in ("DOTNET_TieredCompilation", "COMPlus_TieredCompilation") if key in settings["variables"]]
        require((settings["jit"] == "full-opts" and tiered and all(value == "0" for value in tiered)) or
                (settings["jit"] == "default-tiered" and all(value == "1" for value in tiered)), "JIT label contradicts tiering setting")
        host = parsed["host"]
        for key in ("host", "cpu", "runtime", "isa"):
            require(environment[key] == host[key], "environment/log mismatch: " + key)
        require(int(environment["affinity"], 16) == int(host["affinity"].split()[0], 16), "environment/log affinity mismatch")
        require(set(run["cases"]) == set(CASES), "manifest case identities missing/extra")
        case_ids = {}
        for name, entry in run["cases"].items():
            model = digest(entry["model_sha256"], name + " model_sha256")
            inputs = digest(entry["input_sha256"], name + " input_sha256")
            require(model.startswith(parsed["definitions"][name]["sha12"].lower()), "model/log digest mismatch: " + name)
            case_ids[name] = {"model_sha256": model, "input_sha256": inputs}
            external = entry["external_data"]
            require(isinstance(external, dict), "external_data must be an explicit file/digest map: " + name)
            external_ids = {}
            for location, file_hash in external.items():
                require(isinstance(location, str) and location and not location.startswith(("/", "\\"))
                        and ":" not in location and ".." not in location.replace("\\", "/").split("/"),
                        "invalid relative external tensor location: " + name)
                external_ids[location] = digest(file_hash, name + " external_data")
            case_ids[name]["external_data"] = external_ids
            if name.startswith("e5-"):
                real_tokens = {"e5-8tok": 8, "e5-30tok": 30, "e5-30pad128": 30, "e5-128tok": 128, "e5-512tok": 512}[name]
                require(integer(entry["unmasked_tokens"], "unmasked_tokens", 1) == real_tokens, "E5 attention mask does not match case: " + name)
                case_ids[name]["unmasked_tokens"] = real_tokens
        signature = {"environment": environment, "runner_sha256": digest(run["runner_sha256"], "runner_sha256"),
                     "ort_sha256": digest(native["sha256"], "ORT sha256"), "cases": case_ids,
                     "host": {k: v for k, v in host.items() if k != "lokad"}, "definitions": parsed["definitions"]}
        signatures.append(signature)
        require(signature == signatures[0], label + " workload/host/runner/native identity mismatch")
    if manifest["kind"] == "aa":
        require(identities["L0"] == identities["L1"], "A/A requires identical core binaries and source")
    return {"manifest": str(path), "manifest_sha256": sha256(path), "kind": manifest["kind"],
            "runs": runs, "signature": signatures[0], "identities": identities,
            "start": timestamp(runs[0]["started_utc"]), "end": timestamp(runs[-1]["completed_utc"])}


def load_campaign(path):
    try:
        return _load_campaign(path)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise EvidenceError(str(path) + ": " + str(exc)) from exc
