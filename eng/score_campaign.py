"""Score AMD E5 evidence against a preceding unchanged A/A campaign.

Usage: python eng/score_campaign.py --evidence campaign.json --aa unchanged.json
See eng/campaign-evidence.md for the manifest and acceptance contract.
Historical log-only summaries remain available via eng/parse_baseline.py.

Exit 0: scored PASS/MISS/REGRESSION (inspect verdict, not only the exit code).
Exit 2: invalid/missing evidence. Exit 3: INCONCLUSIVE measurements. Exit 4: usage.
"""
import argparse
import json
import math
import statistics

import campaign_evidence as evidence

PRIMARY = evidence.PRIMARY
SPREAD_CAP = 0.25
HALF_CAP = 0.10
NOISE_CAP = 0.03
PARITY_TARGET = 1.05


def med(xs):
    return statistics.median(xs)


def spread(xs):
    return (max(xs) - min(xs)) / med(xs)


def steady(name, series):
    """Check range and temporal drift; do not sort away sample chronology."""
    if not series or any(not math.isfinite(x) or x <= 0 for x in series):
        return name + ": timings must be finite and positive"
    if spread(series) > SPREAD_CAP:
        return "spread %.3f over %.2f" % (spread(series), SPREAD_CAP)
    n = len(series)
    if n < 2:
        return "too few samples for drift"
    drift = abs(med(series[:n // 2]) - med(series[n // 2:])) / med(series)
    if drift > HALF_CAP:
        return "half-split %.3f over %.2f" % (drift, HALF_CAP)
    return ""


def campaign_stability(campaign, tag, skip=frozenset()):
    problems = []
    for run in campaign["runs"]:
        for name in campaign["cases"]:
            if name in skip:
                continue
            for engine in ("lok", "ort"):
                why = steady(name, run["parsed"]["raw"][name][engine])
                if why:
                    problems.append("%s %s rep%d %s %s: %s" % (tag, run["role"], run["rep"], name, engine, why))
    return problems


def medians(campaign, name):
    pairs = []
    for rep in range(1, 5):
        runs = {run["role"]: run for run in campaign["runs"] if run["rep"] == rep}
        row = {"rep": rep}
        for role in ("L0", "L1"):
            raw = runs[role]["parsed"]["raw"][name]
            row[role] = med(raw["lok"])
            row["O" + role[1]] = med(raw["ort"])
            row["R" + role[1]] = row[role] / row["O" + role[1]]
        pairs.append(row)
    return pairs


def score(campaign, aa):
    evidence.require(campaign["kind"] == "comparison" and aa["kind"] == "aa", "expected comparison and A/A manifests")
    evidence.require((campaign["schema"], campaign["scope"]) == (aa["schema"], aa["scope"]), "A/A evidence schema/scope mismatch")
    cases = campaign["cases"]
    evidence.require(aa["end"] < campaign["start"], "A/A must finish before candidate measurements start")
    def qset(camp):
        out = set()
        for run in camp["runs"]:
            out |= set(run.get("cases_failed", []))
        return out
    aque, cque = qset(aa), qset(campaign)
    quarantined = {name: sorted((["aa"] if name in aque else []) + (["comparison"] if name in cque else []))
                   for name in cases if name in aque or name in cque}
    def sigview(signature):
        return {key: value for key, value in signature.items() if key not in ("cases", "quarantined")}
    evidence.require(sigview(aa["signature"]) == sigview(campaign["signature"]), "A/A workload/host/runner/native identity mismatch")
    common = [name for name in cases if name not in quarantined]
    if campaign["schema"] == 2:
        evidence.require(aa["signature"]["cases"] == campaign["signature"]["cases"],
                         "A/A workload model/input identity mismatch (including quarantined cases)")
    evidence.require({name: aa["signature"]["cases"][name] for name in common} ==
                     {name: campaign["signature"]["cases"][name] for name in common},
                     "A/A workload model/input identity mismatch on measured cases")
    evidence.require(aa["identities"]["L0"] == campaign["identities"]["L0"], "A/A core must match comparison baseline")
    result = {"policy": "amd-e5-v" + str(campaign["schema"]), "scope": campaign["scope"], "campaign_manifest_sha256": campaign["manifest_sha256"],
              "aa_manifest_sha256": aa["manifest_sha256"], "identities": campaign["identities"],
              "jit_regime": campaign["signature"]["environment"]["settings"]["jit"],
              "parity_target": PARITY_TARGET, "noise_cap": NOISE_CAP, "quarantined": quarantined}
    skip = frozenset(quarantined)
    unsteady = campaign_stability(aa, "A/A", skip) + campaign_stability(campaign, "comparison", skip)
    noise = {}
    for name in cases:
        if name in skip:
            continue
        pairs = medians(aa, name)
        # Both arms and their paired ratios; candidate spread never sets a gate.
        controls = {key: spread([pair[key + str(i)] for pair in pairs for i in (0, 1)])
                    for key in ("L", "O", "R")}
        variation = max(controls.values())
        noise[name] = {"variation": variation, "regression_limit": max(0.03, 2 * variation),
                       "ort_limit": max(0.03, 2 * controls["O"]), "spreads": controls}
        if variation > NOISE_CAP:
            unsteady.append("A/A %s variation %.4f exceeds %.2f" % (name, variation, NOISE_CAP))
    result["noise"] = noise
    for name in cases:
        if name in skip:
            continue
        pairs = medians(campaign, name)
        for key in ("L0", "L1", "O0", "O1", "R0", "R1"):
            variation = spread([pair[key] for pair in pairs])
            if variation > NOISE_CAP:
                unsteady.append("comparison %s %s repetition variation %.4f exceeds %.2f" % (name, key, variation, NOISE_CAP))
        for pair in pairs:
            drift = abs(pair["O1"] / pair["O0"] - 1)
            if drift > noise[name]["ort_limit"]:
                unsteady.append("comparison %s rep%d ORT control drift %.4f exceeds %.4f" %
                                (name, pair["rep"], drift, noise[name]["ort_limit"]))
    missing_primary = [name for name in PRIMARY if name in quarantined]
    if missing_primary:
        unsteady.append("primary case quarantined, parity unprovable: " + ",".join(missing_primary))
    if unsteady:
        return dict(result, verdict="INCONCLUSIVE", unsteady=unsteady), 3
    table, regressions = [], []
    for name in common:
        pairs = medians(campaign, name)
        row = {key: med([pair[key] for pair in pairs]) for key in ("L0", "L1", "O0", "O1", "R0", "R1")}
        relative = med([pair["L1"] / pair["L0"] for pair in pairs])
        normalized = med([pair["R1"] / pair["R0"] for pair in pairs])
        threshold = noise[name]["regression_limit"]
        if relative - 1 > threshold or normalized - 1 > threshold:
            regressions.append(name)
        row.update(case=name, reps=pairs, improvement=1 - relative, normalized_improvement=1 - normalized,
                   regression_limit=threshold, gap_closure=(row["R0"] - row["R1"]) / (row["R0"] - 1) if row["R0"] > 1 else None)
        table.append(row)
    for name in cases:
        if name in quarantined:
            table.append({"case": name, "quarantined": quarantined[name]})
    by_name = {row["case"]: row for row in table}
    targets = {name: (name not in quarantined and by_name[name]["R1"] <= PARITY_TARGET) for name in PRIMARY}
    result.update(table=table, targets=targets, regressions=regressions,
                  e5_improvement=(None if missing_primary else 1 - math.prod(1 - by_name[name]["improvement"] for name in PRIMARY) ** 0.25),
                  verdict="REGRESSION" if regressions else ("PASS" if all(targets.values()) else "MISS"))
    return result, 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", help="comparison manifest, schema 1, 2 or isolated e5 schema 3")
    parser.add_argument("--aa", help="preceding unchanged A/A manifest, matching schema and scope")
    # An older lane wrapper must not confuse log-only results with new evidence.
    args, legacy = parser.parse_known_args(argv)
    if legacy or not args.evidence or not args.aa:
        print("campaign-ABORT: release scoring requires --evidence campaign.json --aa unchanged.json; "
              "use eng/parse_baseline.py for historical log-only summaries")
        return 2 if legacy or args.evidence or args.aa else 4
    try:
        result, code = score(evidence.load_campaign(args.evidence), evidence.load_campaign(args.aa))
    except evidence.EvidenceError as exc:
        print("campaign-ABORT: " + str(exc))
        return 2
    if "table" in result:
        print("| Case | L0 ms | L1 ms | ORT0 ms | ORT1 ms | paired R0 | paired R1 | improvement |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in result["table"]:
            if "quarantined" in row:
                print("| " + row["case"] + " | quarantined (" + ",".join(row["quarantined"]) + ") |")
            else:
                print("| {case} | {L0:.3f} | {L1:.3f} | {O0:.3f} | {O1:.3f} | {R0:.3f} | {R1:.3f} | {improvement:.2%} |".format(**row))
    print("campaign-" + result["verdict"] + " (measurement verdict; not a ship decision)")
    print(json.dumps(result, indent=1, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
