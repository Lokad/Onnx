"""Supervise schema-3/4 e5 evidence with separate inference engines.

Prepare with build-common-campaign.ps1. A/A uses the release core in both roles;
comparison requires qualified matching --aa evidence from this exact producer.
See isolated-evidence.md. No remote operations, asset downloads or overwrites.
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
import isolated_evidence as isolated
import run_common_campaign as common
import score_campaign


def child(build, output, role, rep, mode, name, fixture, cpu, smoke, env):
    return _child(build, output, role, rep, mode, name, fixture, cpu, smoke, env, isolated.PROTOCOL)


def child_conditioned(build, output, role, rep, mode, name, fixture, cpu, smoke, env):
    return _child(build, output, role, rep, mode, name, fixture, cpu, smoke, env, isolated.CONDITIONED_PROTOCOL)


def _child(build, output, role, rep, mode, name, fixture, cpu, smoke, env, protocol):
    stem = "%s-rep%d-%s-%s" % (role.lower(), rep, name, mode)
    paths = {key: output / (stem + suffix) for key, suffix in
             (("process", ".process.json"), ("supervision", ".supervision.json"), ("config", ".config.json"),
              ("log", ".log"), ("error", ".err.log"), ("before", ".pre.json"), ("after", ".post.json"))}
    config = dict(root=str(common.ROOT), case=name, cpu=cpu, output=str(paths["process"]), source_sha=build["source_sha"].lower(),
                  core_sha256=build["core_sha256"].lower(), fixture=str(fixture),
                  fixture_sha256=None if mode == "oracle" else evidence.sha256(fixture / "fixture.json"), smoke=smoke)
    common.save_json(paths["config"], config)
    command = ["dotnet", str(build["runner"]), isolated.PROTOCOLS[protocol][2], mode, str(paths["config"])]
    before = processes.snapshot()
    common.save_json(paths["before"], before)
    launched = dt.datetime.now(dt.timezone.utc)
    print("Starting " + stem + (" (unscored smoke)" if smoke else ""), flush=True)
    failure = None
    with paths["log"].open("x", encoding="utf-8") as stdout, paths["error"].open("x", encoding="utf-8") as stderr:
        process = common.spawn_child(command, cpu, cwd=common.ROOT, stdout=stdout, stderr=stderr, env=env)
        try:
            code = common.wait_child(process, stem, timeout=600)
        except BaseException as exc:
            # wait_child stops only this owned child. Preserve accounting and
            # observed exit even when a timeout/cancellation interrupts waiting.
            failure = exc
            code = process.returncode
        exited = dt.datetime.now(dt.timezone.utc)
    after = processes.snapshot()
    common.save_json(paths["after"], after)
    accounting = processes.foreign_fraction(before, after, os.getpid())
    common.save_json(paths["supervision"], dict(supervisor_pid=os.getpid(), pid=process.pid, mode=mode, case=name,
                     launched_utc=launched.isoformat(), exited_utc=exited.isoformat(), exit_code=code,
                     command=command, accounting=accounting))
    if failure is not None:
        raise failure
    evidence.require(code == 0, "child failed: " + str(paths["error"]))
    entry = {key: isolated.binding(path, output) for key, path in paths.items()}
    record = isolated.validate_process_for_protocol(output, entry, smoke, protocol)
    evidence.require(record["source_sha"] == config["source_sha"] and record["core_sha256"] == config["core_sha256"], "worker differs from staged core")
    evidence.require(record["environment"]["sdk"] == build["sdk"], "core/runner SDK mismatch")
    print("Finished %s foreign_cpu=%.4f" % (stem, accounting["foreign_cpu_fraction"]), flush=True)
    return entry, record


def verify_oracle_against_aa(record, case_identity, aa):
    if aa is None:
        return
    signature = aa["signature"]
    for key in ("environment", "runner_sha256", "ort_version", "oracle_native_sha256", "execution", "timing_contract"):
        evidence.require(record[key] == signature[key], "A/A preflight identity differs: " + key)
    evidence.require(case_identity == signature["cases"][record["case"]], "A/A preflight workload/oracle identity differs")
    evidence.require((record["source_sha"], record["core_sha256"]) == tuple(aa["identities"]["L0"]), "A/A baseline core differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--kind", choices=("aa", "comparison"), required=True)
    parser.add_argument("--aa", type=Path)
    parser.add_argument("--jit", choices=("default-tiered", "full-opts"), default="default-tiered")
    parser.add_argument("--conditioning", choices=("none", "30s"), default="none",
                        help="30s selects schema 4 with retained inference conditioning; none preserves schema 3")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--root", type=Path, default=common.ROOT, help="repository with local models; defaults to this checkout")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    protocol = isolated.CONDITIONED_PROTOCOL if args.conditioning == "30s" else isolated.PROTOCOL
    schema = isolated.PROTOCOLS[protocol][0]
    launch_child = child_conditioned if args.conditioning == "30s" else child
    try:
        evidence.require(0 <= args.cpu < 64, "invalid CPU selector")
        evidence.require(args.kind != "comparison" or args.smoke or args.aa is not None, "comparison requires qualified --aa")
        evidence.require(args.kind != "aa" or args.aa is None, "A/A cannot consume another calibration")
        aa = evidence.load_campaign(args.aa) if args.aa else None
        if aa is not None:
            evidence.require(aa["schema"] == schema, "A/A conditioning protocol/schema differs before launch")
            health = isolated.calibration_health(aa)
            evidence.require(health["qualified"], "A/A failed unchanged noise gates: " + json.dumps(health["problems"]))
        common.ROOT = args.root.resolve()
        script_directory = Path(__file__).resolve().parent
        script_hashes = {name: evidence.sha256(script_directory / name) for name in isolated.SCRIPTS}
        if aa is not None:
            evidence.require(script_hashes == aa["signature"]["producer_files"], "A/A supervisor/scorer sources differ before launch")
        for asset in common.ASSETS["e5"]:
            evidence.require((common.ROOT / "models" / asset).is_file(), "local model asset missing: " + asset)
        env = os.environ.copy()
        for key in list(env):
            if key.lower() in ("dotnet_tieredcompilation", "complus_tieredcompilation"):
                del env[key]
        env["DOTNET_TieredCompilation"] = "0" if args.jit == "full-opts" else "1"
        builds = common.stage(args.prepared.resolve(), output, args.kind)
        (output / "producer").mkdir()
        for name in isolated.SCRIPTS:
            shutil.copyfile(script_directory / name, output / "producer" / name)
        manifest = dict(schema=schema, scope="e5", protocol=protocol, kind="smoke" if args.smoke else args.kind,
                        cooldown_seconds=0 if args.smoke else 300, oracles=[], runs=[],
                        producer_files={name: isolated.binding(output / "producer" / name, output) for name in isolated.SCRIPTS},
                        builds={role: {key: build[key] for key in ("source_sha", "core_sha256", "source_archive_sha256", "sdk")} for role, build in builds.items()})
        fixtures = {}
        reference = None
        for name in isolated.CASES:
            fixture = output / (name + "-oracle")
            fixtures[name] = fixture
            entry, record = launch_child(builds["L0"], output, "L0", 0, "oracle", name, fixture, args.cpu, args.smoke, env)
            fixture_binding = isolated.binding(fixture / "fixture.json", output)
            case_identity = isolated.validate_fixture_for_protocol(output, fixture_binding, record, protocol)
            verify_oracle_against_aa(record, case_identity, aa)
            manifest["oracles"].append(dict(worker=entry, fixture=fixture_binding))
            if reference is None:
                reference = record
            for key in ("environment", "runner_sha256", "oracle_native_sha256"):
                evidence.require(record[key] == reference[key], "oracle identity changed: " + key)
        # Check both staged identities before beginning the long measured visits.
        # These probes initialize ORT but contain no inference/model timings.
        # They use the unchanged legacy probe producer solely for actual binary
        # discovery. They are never included among isolated observations.
        common.preflight(builds, output, args.cpu, env, None, "e5")
        for role in ("L0", "L1"):
            probe = isolated.read_json(output / (role.lower() + "-probe.process.json"))
            evidence.require(probe["environment"] == reference["environment"] and probe["runner_sha256"] == reference["runner_sha256"]
                             and probe["ort_native"]["sha256"] == reference["oracle_native_sha256"], "staged preflight differs from oracle")
        order = evidence.ORDER[:2] if args.smoke else evidence.ORDER
        for index, role in enumerate(order):
            if index > 0 and index % 2 == 0:
                for remaining in range(300, 0, -30):
                    print("Pair cooldown: %ds" % remaining, flush=True)
                    time.sleep(30)
            visit = dict(role=role, rep=index // 2 + 1, workers=[])
            engines = ("lok", "ort") if index % 2 == 0 else ("ort", "lok")
            for name in isolated.CASES:
                for mode in engines:
                    entry, record = launch_child(builds[role], output, role, visit["rep"], mode, name, fixtures[name], args.cpu, args.smoke, env)
                    for key in ("environment", "runner_sha256", "oracle_native_sha256"):
                        evidence.require(record[key] == reference[key], "worker identity changed: " + key)
                    visit["workers"].append(entry)
            manifest["runs"].append(visit)
        destination = output / "evidence.json"
        evidence.require(all(evidence.sha256(script_directory / name) == digest for name, digest in script_hashes.items()), "producer sources changed during campaign")
        common.save_json(destination, manifest)
        campaign = isolated.load(destination, manifest, allow_smoke=args.smoke)
        if args.smoke:
            print("SMOKE COMPLETE: " + str(destination) + "; no performance verdict")
            return 0
        if args.kind == "aa":
            health = isolated.calibration_health(campaign)
            common.save_json(output / "calibration-health.json", health)
            print("A/A " + ("QUALIFIED" if health["qualified"] else "INCONCLUSIVE") + ": " + str(destination))
            print(json.dumps(health, indent=2))
            return 0 if health["qualified"] else 3
        return score_campaign.main(["--evidence", str(destination), "--aa", str(args.aa.resolve())])
    except (evidence.EvidenceError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print("isolated-campaign-ABORT: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
