"""Supervise the identical runner against pinned isolated cores, locally.

Prepare with eng/build-common-campaign.ps1, then --prepared DIR --output NEWDIR
--cpu N --kind aa, or --kind comparison --aa prior/evidence.json. --smoke
uses one pair and tiny counts, and always writes kind=smoke (never release).
No SSH, fetching, worktree removal or modification of existing outputs.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import campaign_evidence as evidence
import campaign_processes as processes
import score_campaign

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "Lokad.Onnx.Campaign.dll"
ASSETS = {
    "e5": ["multilingual-e5-small/model.onnx", "multilingual-e5-small/sentencepiece.bpe.model"],
    "dinov3": ["dinov3-vits16/onnx/model.onnx", "dinov3-vits16/onnx/model.onnx_data"],
    "resnet50": ["resnet50-onnx/model.onnx"],
    "gpt2": ["gpt2-onnx/onnx/model.onnx"],
}


def save_json(path, value):
    # All output paths are newly owned files. Never overwrite prior evidence.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_build(prepared, role):
    path = prepared / role / "build.json"
    build = json.loads(path.read_text(encoding="utf-8-sig"))
    core = (path.parent / build["core"]).resolve()
    evidence.require(core.is_file(), "missing core: " + str(core))
    evidence.require(evidence.digest(build["core_sha256"], "built core") == evidence.sha256(core), "built core changed: " + role)
    evidence.require(evidence.digest(build["source_archive_sha256"], "source archive") == evidence.sha256(path.parent / "source.zip"), "source archive changed: " + role)
    evidence.digest(build["source_sha"], "source_sha", 40)
    evidence.require(isinstance(build["sdk"], str) and build["sdk"], "build SDK missing")
    return dict(build, path=core)


def stage(prepared, output, kind):
    baseline = read_build(prepared, "L0")
    candidate = read_build(prepared, "L1") if kind == "comparison" else baseline
    evidence.require(baseline["sdk"] == candidate["sdk"], "core build SDKs differ")
    runner = prepared / "runner"
    evidence.require((runner / RUNNER).is_file(), "common runner missing")
    evidence.require(not output.exists(), "output already exists; choose a fresh directory")
    evidence.require(not output.is_relative_to(prepared), "output must be outside the prepared tree")
    output.mkdir(parents=True)
    result = {}
    for role, build in (("L0", baseline), ("L1", candidate)):
        destination = output / "staged" / role
        shutil.copytree(runner, destination)
        shutil.copyfile(build["path"], destination / "Lokad.Onnx.dll")
        result[role] = dict(build, runner=destination / RUNNER)
    return result


def finish_record(path, build, role, rep, pid, launched, exited, code, log, accounting):
    evidence.require(code == 0, "child failed; inspect " + str(log.with_suffix(".err.log")))
    record = json.loads(path.read_text(encoding="utf-8"))
    evidence.require(record["producer"] == "common-runner-v1", "wrong evidence producer")
    evidence.require(record["process_id"] == pid and record["exit_code"] == code, "child process/exit identity mismatch")
    evidence.require(record["core_sha256"] == build["core_sha256"].lower() and record["source_sha"] == build["source_sha"].lower(), "loaded core/source differs from staged build")
    evidence.require(record["environment"]["sdk"] == build["sdk"], "runner and core SDKs differ")
    start, end = evidence.timestamp(record["started_utc"]), evidence.timestamp(record["completed_utc"])
    evidence.require(launched <= start <= end <= exited,
                     "child timestamps outside supervised interval: launched=" + launched.isoformat()
                     + " start=" + record["started_utc"] + " end=" + record["completed_utc"]
                     + " exited=" + exited.isoformat())
    record.update(role=role, rep=rep, log=log.name, log_sha256=evidence.sha256(log), accounting=accounting,
                  completed_utc=exited.isoformat(), process_evidence=path.name,
                  process_evidence_sha256=evidence.sha256(path), source_archive_sha256=build["source_archive_sha256"])
    return record


def windows_affinity():
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.GetProcessAffinityMask.restype = ctypes.c_int
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel.SetProcessAffinityMask.restype = ctypes.c_int
    handle = kernel.GetCurrentProcess()
    current, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel, handle, current.value


def spawn_child(command, cpu, **kwargs):
    # Pin before the CLR starts: runtime workers must inherit the same CPU too.
    if sys.platform.startswith("linux"):
        return subprocess.Popen(["taskset", "--cpu-list", str(cpu), *command], **kwargs)
    evidence.require(sys.platform == "win32", "launch-time affinity requires Windows or Linux")
    import ctypes
    kernel, handle, original = windows_affinity()
    requested = 1 << cpu
    evidence.require(bool(original & requested), "requested CPU is outside the supervisor's allowed affinity")
    if not kernel.SetProcessAffinityMask(handle, requested):
        raise ctypes.WinError(ctypes.get_last_error())
    child = None
    try:
        # Windows children inherit the parent's affinity. Only this supervisor
        # is briefly narrowed, and its original mask is restored immediately.
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        child = subprocess.Popen(command, **kwargs)
        return child
    finally:
        if not kernel.SetProcessAffinityMask(handle, original):
            failure = ctypes.WinError(ctypes.get_last_error())
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()
            raise failure


def wait_child(child, label, timeout=3600):
    started = time.monotonic()
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(label + " exceeded its process time limit")
            try:
                return child.wait(timeout=min(30, remaining))
            except subprocess.TimeoutExpired:
                print(label + " still running", flush=True)
    except BaseException:
        # This Popen object belongs only to the child we launched. Never search
        # for or terminate other agents' processes by name/PID enumeration.
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise


def preflight(builds, output, cpu, child_env, aa):
    records = []
    for role, build in builds.items():
        stem = role.lower() + "-probe"
        log, error, record_path = (output / (stem + suffix) for suffix in (".log", ".err.log", ".process.json"))
        command = ["dotnet", str(build["runner"]), "e5", "--probe", "--cpu", str(cpu),
                   "--evidence-out", str(record_path), "--source-sha", build["source_sha"],
                   "--expected-core-sha256", build["core_sha256"]]
        start = dt.datetime.now(dt.timezone.utc)
        with log.open("x", encoding="utf-8") as stdout, error.open("x", encoding="utf-8") as stderr:
            child = spawn_child(command, cpu, cwd=ROOT, stdout=stdout, stderr=stderr, env=child_env)
            code = wait_child(child, stem, timeout=120)
        record = finish_record(record_path, build, role, 0, child.pid, start, dt.datetime.now(dt.timezone.utc), code, log,
                               {"valid": False, "reason": "probe has no measured workload or accounting interval"})
        records.append(record)
    left, right = records
    for key in ("runner_sha256", "environment"):
        evidence.require(left[key] == right[key], "preflight differs between cores: " + key)
    evidence.require(left["ort_native"]["sha256"] == right["ort_native"]["sha256"], "preflight ORT binaries differ")
    if aa:
        signature = aa["signature"]
        evidence.require(tuple(aa["identities"]["L0"]) == (left["source_sha"], left["core_sha256"]), "A/A baseline differs before launch")
        for key in ("runner_sha256", "environment"):
            evidence.require(left[key] == signature[key], "A/A preflight mismatch: " + key)
        evidence.require(left["ort_native"]["sha256"] == signature["ort_sha256"], "A/A ORT module differs before launch")
        hashes = {}
        for name, case in signature["cases"].items():
            model = next(key for key in ASSETS if name.startswith(key + "-") or key == "resnet50" and name.startswith("resnet50-"))
            path = ROOT / "models" / ASSETS[model][0]
            if path not in hashes:
                hashes[path] = evidence.sha256(path)
            evidence.require(hashes[path] == case["model_sha256"], "A/A model changed before launch: " + name)
            for relative, expected in case["external_data"].items():
                external = path.parent / relative
                if external not in hashes:
                    hashes[external] = evidence.sha256(external)
                evidence.require(hashes[external] == expected, "A/A external tensor changed before launch: " + relative)
    print("PREFLIGHT OK: identical runner, runtime and native ORT; expected cores loaded", flush=True)


def run_leg(build, role, rep, output, cpu, models, only_case, smoke, child_env):
    stem = "%s-rep%d" % (role.lower(), rep)
    log, error, record_path = (output / (stem + suffix) for suffix in (".log", ".err.log", ".process.json"))
    args = ["dotnet", str(build["runner"]), *models, "--cpu", str(cpu), "--iters", "2" if smoke else "33",
            "--evidence-out", str(record_path), "--source-sha", build["source_sha"], "--expected-core-sha256", build["core_sha256"]]
    if smoke:
        args.extend(["--warmup", "1", "--warmup-min", "-1", "--warmup-max", "-1", "--warmup-ms", "0"])
    if only_case:
        args.extend(["--case", only_case])
    before = processes.snapshot()
    save_json(output / (stem + ".pre.json"), before)
    launched = dt.datetime.now(dt.timezone.utc)
    print("Starting %s: %s" % (stem, "smoke, not release evidence" if smoke else "33 samples, duration/convergence warmup"), flush=True)
    with log.open("x", encoding="utf-8") as stdout, error.open("x", encoding="utf-8") as stderr:
        child = spawn_child(args, cpu, cwd=ROOT, stdout=stdout, stderr=stderr, env=child_env)
        code = wait_child(child, stem)
        exited = dt.datetime.now(dt.timezone.utc)
    after = processes.snapshot()
    save_json(output / (stem + ".post.json"), after)
    accounting = processes.foreign_fraction(before, after, os.getpid())
    save_json(output / (stem + ".supervision.json"), {"pid": child.pid, "launched_utc": launched.isoformat(),
              "exited_utc": exited.isoformat(), "exit_code": code, "command": args, "accounting": accounting})
    record = finish_record(record_path, build, role, rep, child.pid, launched, exited, code, log, accounting)
    evidence.require(smoke or accounting["foreign_cpu_fraction"] <= 0.10, "foreign CPU contamination; retained diagnostic evidence")
    print("Finished %s exit=%d foreign_cpu=%.4f" % (stem, code, accounting["foreign_cpu_fraction"]), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu", required=True, type=int)
    parser.add_argument("--kind", required=True, choices=("aa", "comparison"))
    parser.add_argument("--aa", type=Path)
    parser.add_argument("--jit", choices=("default-tiered", "full-opts"), default="default-tiered")
    parser.add_argument("--cooldown", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--models", nargs="+", choices=tuple(ASSETS), default=list(ASSETS))
    parser.add_argument("--case")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        evidence.require(0 <= args.cpu < min(64, os.cpu_count() or 1), "CPU selector is outside the supported mask")
        evidence.require(args.cooldown >= 0, "negative cooldown")
        evidence.require(args.smoke or args.models == list(ASSETS) and args.case is None, "release requires all canonical models/cases")
        evidence.require(args.kind != "comparison" or args.smoke or args.aa is not None, "comparison requires preceding --aa evidence")
        aa = evidence.load_campaign(args.aa) if args.aa else None  # Before any measurement work.
        if aa:
            evidence.require(aa["kind"] == "aa", "--aa is not an unchanged campaign")
        for model in args.models:
            for file in ASSETS[model]:
                evidence.require((ROOT / "models" / file).is_file(), "missing local asset: " + file)
        child_env = os.environ.copy()
        # Set both aliases consistently; do not alter the supervisor environment.
        for key in list(child_env):
            if key.lower() in ("dotnet_tieredcompilation", "complus_tieredcompilation"):
                del child_env[key]
        child_env["DOTNET_TieredCompilation"] = "0" if args.jit == "full-opts" else "1"
        builds = stage(args.prepared.resolve(), output, args.kind)
        preflight(builds, output, args.cpu, child_env, aa)
        manifest = {"schema": 1, "kind": "smoke" if args.smoke else args.kind, "runs": []}
        order = evidence.ORDER[:2] if args.smoke else evidence.ORDER
        for i, role in enumerate(order):
            if i > 0 and i % 2 == 0:
                for remaining in range(args.cooldown, 0, -30):
                    print("Cooldown: %ds" % remaining, flush=True)
                    time.sleep(min(30, remaining))
            record = run_leg(builds[role], role, i // 2 + 1, output, args.cpu, args.models, args.case, args.smoke, child_env)
            manifest["runs"].append(record)
        destination = output / "evidence.json"
        save_json(destination, manifest)
        if args.smoke:
            # Check cross-core identities even though smoke timings cannot score.
            left, right = manifest["runs"]
            for key in ("runner_sha256", "cases", "environment"):
                evidence.require(left[key] == right[key], "smoke cross-core identity differs: " + key)
            evidence.require(left["ort_native"]["sha256"] == right["ort_native"]["sha256"], "smoke ORT binary differs")
            print("SMOKE COMPLETE: " + str(destination) + " (not release evidence)")
            return 0
        evidence.load_campaign(destination)
        if args.kind == "aa":
            print("A/A CAPTURED: " + str(destination) + "; noise/stability will be checked when scoring a comparison")
            return 0
        return score_campaign.main(["--evidence", str(destination), "--aa", str(args.aa.resolve())])
    except (evidence.EvidenceError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print("campaign-ABORT: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
