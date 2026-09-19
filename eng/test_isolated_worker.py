"""Live, explicitly unscored smoke for the isolated e5 producer.

Requires local e5 assets and a common runner compiled against the frozen release
API. Uses launch-time CPU confinement and fresh processes for oracle/Lokad/ORT.
Output must be a new directory; logs and negative-case failures are preserved.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import campaign_evidence as evidence
import isolated_evidence as isolated
import run_common_campaign as common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cpu", required=True, type=int)
    parser.add_argument("--cases", nargs="+", choices=evidence.CASES[:5], default=list(evidence.CASES[:5]))
    parser.add_argument("--conditioning", choices=("none", "30s"), default="none")
    args = parser.parse_args(argv)
    runner, output = args.runner.resolve(), args.output.resolve()
    protocol = isolated.CONDITIONED_PROTOCOL if args.conditioning == "30s" else isolated.PROTOCOL
    _, contract, verb = isolated.PROTOCOLS[protocol]
    evidence.digest(args.source, "source", 40)
    evidence.require(not output.exists(), "output already exists")
    output.mkdir(parents=True)
    core_hash = evidence.sha256(runner.parent / "Lokad.Onnx.dll")
    env = os.environ.copy()
    env["DOTNET_TieredCompilation"] = "0"
    # Keep any already-selected candidate flags: they are recorded by the worker.
    configs, records = {}, {}

    def launch(mode, label, config, rejection=None):
        config_path = output / (label + ".config.json")
        common.save_json(config_path, config)
        with (output / (label + ".log")).open("x", encoding="utf-8") as log, (output / (label + ".err.log")).open("x", encoding="utf-8") as error:
            child = common.spawn_child(["dotnet", str(runner), verb, mode, str(config_path)], args.cpu,
                                       cwd=common.ROOT, stdout=log, stderr=error, env=env)
            code = common.wait_child(child, label)
        if rejection:
            assert code != 0 and not Path(config["output"]).exists(), label
            assert rejection in (output / (label + ".err.log")).read_text(encoding="utf-8"), label
            return None
        assert code == 0, label + ": see preserved error log"
        record = json.loads(Path(config["output"]).read_text(encoding="utf-8"))
        assert record["producer"] == protocol and record["timing_contract"] == contract and record["core_sha256"] == core_hash
        assert record["configuration_sha256"] == evidence.sha256(config_path)
        assert record["mode"] == mode and record["case"] == config["case"] and record["smoke"] is True
        assert record["process_id"] == child.pid and record["inputs_intact"] is True
        assert int(record["environment"]["affinity"], 16) == 1 << args.cpu
        assert bool(record["native_module"]) == (mode != "lok")
        isolated._validate_measurements(record, True, protocol)
        if mode != "oracle":
            measured = record["measured"]
            assert 0 <= measured["pre_scaled_error"] <= 1e-4 and 0 <= measured["post_scaled_error"] <= 1e-4
            assert len(measured["warmup_ticks"]) == 1 and measured["warmup_stop"] == "fixed-smoke"
            assert len(measured["blocks"]) == 1
            block = measured["blocks"][0]
            assert len(block["execute_ticks"]) == 2 and all(t > 0 for t in block["execute_ticks"])
            assert sum(block["execute_ticks"]) <= block["block_ticks"]
        print(label + " passed", flush=True)
        return record

    for name in args.cases:
        fixture = output / (name + "-oracle")
        config = dict(root=str(common.ROOT), case=name, cpu=args.cpu, output=str(output / (name + "-oracle.process.json")),
                      source_sha=args.source, core_sha256=core_hash, fixture=str(fixture), fixture_sha256=None, smoke=True)
        oracle = launch("oracle", name + "-oracle", config)
        config["fixture_sha256"] = evidence.sha256(fixture / "fixture.json")
        workers = []
        for mode in ("lok", "ort"):
            config["output"] = str(output / (name + "-" + mode + ".process.json"))
            workers.append(launch(mode, name + "-" + mode, config))
        for field in ("fixture_sha256", "model_sha256", "tokenizer_sha256", "input_sha256", "environment", "runner_sha256", "oracle_native_sha256"):
            assert oracle[field] == workers[0][field] == workers[1][field], name + ": " + field
        configs[name] = config.copy()
        records[name] = [oracle, *workers]

    original = configs[args.cases[0]]
    for label, changes, rejection in (
        ("wrong-core", {"core_sha256": "0" * 64}, "Loaded core differs"),
        ("wrong-fixture", {"fixture_sha256": "0" * 64}, "Oracle manifest digest differs"),
        ("wrong-case", {"case": "not-e5"}, "Unknown isolated e5 case"),
        ("invalid-cpu", {"cpu": 64}, "Invalid CPU selector"),
    ):
        config = dict(original, output=str(output / (label + ".process.json")), **changes)
        launch("lok", label, config, rejection)
    corrupt = output / "corrupt-oracle"
    shutil.copytree(original["fixture"], corrupt)
    (corrupt / "output-0.f32").write_bytes(b"bad")
    launch("lok", "corrupt-oracle", dict(original, fixture=str(corrupt), output=str(output / "corrupt.process.json")), "Oracle output bytes/digest differ")
    missing = output / "missing-outputs"
    shutil.copytree(original["fixture"], missing)
    manifest = json.loads((missing / "fixture.json").read_text(encoding="utf-8"))
    manifest["outputs"] = []
    (missing / "fixture.json").write_text(json.dumps(manifest), encoding="utf-8")
    launch("lok", "missing-outputs", dict(original, fixture=str(missing), fixture_sha256=evidence.sha256(missing / "fixture.json"),
        output=str(output / "missing.process.json")), "Oracle outputs missing")
    mixed = output / "mixed-protocol"
    shutil.copytree(original["fixture"], mixed)
    manifest = json.loads((mixed / "fixture.json").read_text(encoding="utf-8"))
    manifest["protocol"] = isolated.PROTOCOL if protocol == isolated.CONDITIONED_PROTOCOL else isolated.CONDITIONED_PROTOCOL
    (mixed / "fixture.json").write_text(json.dumps(manifest), encoding="utf-8")
    launch("lok", "mixed-protocol", dict(original, fixture=str(mixed), fixture_sha256=evidence.sha256(mixed / "fixture.json"),
        output=str(output / "mixed.process.json")), "Oracle workload identity differs")
    # Existing outputs are a refusal, not a retry that rewrites evidence.
    label = "existing-output"
    config_path = output / (label + ".config.json")
    common.save_json(config_path, original)
    completed_hash = evidence.sha256(original["output"])
    with (output / (label + ".log")).open("x", encoding="utf-8") as log, (output / (label + ".err.log")).open("x", encoding="utf-8") as error:
        child = common.spawn_child(["dotnet", str(runner), verb, "lok", str(config_path)], args.cpu, cwd=common.ROOT, stdout=log, stderr=error, env=env)
        assert common.wait_child(child, label) != 0
    assert "Process output already exists" in (output / (label + ".err.log")).read_text(encoding="utf-8")
    assert evidence.sha256(original["output"]) == completed_hash
    common.save_json(output / "summary.json", {"scope": "unscored live smoke", "protocol": protocol, "cases": args.cases,
                                               "core_sha256": core_hash, "negative_checks": 8, "records": records})
    print("Isolated worker smoke and eight refusal checks passed; no timing verdict.")


if __name__ == "__main__":
    main()
