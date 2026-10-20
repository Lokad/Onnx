"""Separate e5 worker launch and validation, independent of campaign scoring."""
import datetime as dt
import os

import benchmark_worker as common
import campaign_evidence as evidence
import campaign_processes as processes
import isolated_evidence as isolated

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
