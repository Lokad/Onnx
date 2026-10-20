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
import subprocess
import sys
import time

import campaign_evidence as evidence
import campaign_processes as processes
import score_campaign
from benchmark_worker import RUNNER, save_json, read_build, stage, windows_affinity, spawn_child, wait_child

ROOT = Path(__file__).resolve().parent.parent
ASSETS = {
    "e5": ["multilingual-e5-small/model.onnx", "multilingual-e5-small/sentencepiece.bpe.model"],
    "dinov3": ["dinov3-vits16/onnx/model.onnx", "dinov3-vits16/onnx/model.onnx_data"],
    "resnet50": ["resnet50-onnx/model.onnx"],
    "gpt2": ["gpt2-onnx/onnx/model.onnx"],
}


def finish_record(path, build, role, rep, pid, launched, exited, code, log, accounting, scope=None):
    evidence.require(code == 0, "child failed; inspect " + str(log.with_suffix(".err.log")))
    record = json.loads(path.read_text(encoding="utf-8"))
    evidence.require(record["producer"] == ("common-runner-v1" if scope is None else "common-runner-v2"), "wrong evidence producer")
    if scope is not None:
        evidence.require(record["scope"] == scope, "child evidence scope differs from requested scope")
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


def preflight(builds, output, cpu, child_env, aa, scope, root=None):
    root = ROOT if root is None else root
    records = []
    for role, build in builds.items():
        stem = role.lower() + "-probe"
        log, error, record_path = (output / (stem + suffix) for suffix in (".log", ".err.log", ".process.json"))
        command = ["dotnet", str(build["runner"]), "e5", "--probe", "--cpu", str(cpu),
                   "--evidence-out", str(record_path), "--source-sha", build["source_sha"],
                   "--expected-core-sha256", build["core_sha256"], "--evidence-scope", scope]
        start = dt.datetime.now(dt.timezone.utc)
        with log.open("x", encoding="utf-8") as stdout, error.open("x", encoding="utf-8") as stderr:
            child = spawn_child(command, cpu, cwd=root, stdout=stdout, stderr=stderr, env=child_env)
            code = wait_child(child, stem, timeout=120)
        record = finish_record(record_path, build, role, 0, child.pid, start, dt.datetime.now(dt.timezone.utc), code, log,
                               {"valid": False, "reason": "probe has no measured workload or accounting interval"}, scope)
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
            path = root / "models" / ASSETS[model][0]
            if path not in hashes:
                hashes[path] = evidence.sha256(path)
            evidence.require(hashes[path] == case["model_sha256"], "A/A model changed before launch: " + name)
            for relative, expected in case["external_data"].items():
                external = path.parent / relative
                if external not in hashes:
                    hashes[external] = evidence.sha256(external)
                evidence.require(hashes[external] == expected, "A/A external tensor changed before launch: " + relative)
    print("PREFLIGHT OK: identical runner, runtime and native ORT; expected cores loaded", flush=True)


def run_leg(build, role, rep, output, cpu, models, only_case, smoke, child_env, scope):
    stem = "%s-rep%d" % (role.lower(), rep)
    log, error, record_path = (output / (stem + suffix) for suffix in (".log", ".err.log", ".process.json"))
    args = ["dotnet", str(build["runner"]), *models, "--cpu", str(cpu), "--iters", "2" if smoke else "33",
            "--evidence-out", str(record_path), "--source-sha", build["source_sha"], "--expected-core-sha256", build["core_sha256"],
            "--evidence-scope", scope]
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
    record = finish_record(record_path, build, role, rep, child.pid, launched, exited, code, log, accounting, scope)
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
    parser.add_argument("--scope", choices=tuple(evidence.SCOPES), default="full")
    parser.add_argument("--models", nargs="+", choices=tuple(ASSETS), help="smoke only; scored workloads are selected by --scope")
    parser.add_argument("--case")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        evidence.require(0 <= args.cpu < min(64, os.cpu_count() or 1), "CPU selector is outside the supported mask")
        evidence.require(args.cooldown >= 0, "negative cooldown")
        evidence.require(args.smoke or args.models is None and args.case is None, "scored workloads require --scope, without --models or --case")
        models = args.models if args.models is not None else (["e5"] if args.scope == "e5" else list(ASSETS))
        evidence.require(args.kind != "comparison" or args.smoke or args.aa is not None, "comparison requires preceding --aa evidence")
        aa = evidence.load_campaign(args.aa) if args.aa else None  # Before any measurement work.
        if aa:
            evidence.require(aa["kind"] == "aa", "--aa is not an unchanged campaign")
            evidence.require(aa["schema"] == 2 and aa["scope"] == args.scope, "A/A evidence schema/scope mismatch before launch")
        for model in models:
            for file in ASSETS[model]:
                evidence.require((ROOT / "models" / file).is_file(), "missing local asset: " + file)
        child_env = os.environ.copy()
        # Set both aliases consistently; do not alter the supervisor environment.
        for key in list(child_env):
            if key.lower() in ("dotnet_tieredcompilation", "complus_tieredcompilation"):
                del child_env[key]
        child_env["DOTNET_TieredCompilation"] = "0" if args.jit == "full-opts" else "1"
        builds = stage(args.prepared.resolve(), output, args.kind)
        preflight(builds, output, args.cpu, child_env, aa, args.scope)
        manifest = {"schema": 2, "scope": args.scope, "kind": "smoke" if args.smoke else args.kind, "runs": []}
        order = evidence.ORDER[:2] if args.smoke else evidence.ORDER
        for i, role in enumerate(order):
            if i > 0 and i % 2 == 0:
                for remaining in range(args.cooldown, 0, -30):
                    print("Cooldown: %ds" % remaining, flush=True)
                    time.sleep(min(30, remaining))
            record = run_leg(builds[role], role, i // 2 + 1, output, args.cpu, models, args.case, args.smoke, child_env, args.scope)
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
