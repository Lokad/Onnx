"""Score a matched L0/L1/ORT campaign (E01/E02 lane).

Usage: python eng/score_campaign.py L0REP... --l1 L1REP...
Each rep log is one fresh Bench process in the canonical rows format read by
eng/parse_baseline.py. The lok series is the Lokad public-Execute boundary and
the ort series is the ORT boundary; ctx is carried for diagnosis and is never
gated or scored.

Exit codes: 0 scored (verdict inside the JSON, may be PASS, MISS, or
REGRESSION); 2 host or manifest mismatch between reps (ABORT, retry the
campaign); 3 unsteady series (INCONCLUSIVE, no score); 4 usage error or fewer
than 33 samples per rep or fewer than 3 reps per library.

Gates (provisional, recorded here): per case, engine, and rep, the raw series
must satisfy spread (max-min)/median <= 0.25 and half-split
|median(first)-median(second)|/median <= 0.10 for both lok and ort.
"""
import json
import statistics
import sys

import parse_baseline as base

PRIMARY = base.ORDER
SPREAD_CAP = 0.25
HALF_CAP = 0.10
MIN_SAMPLES = 33
MIN_REPS = 3
HOST_IGNORE = {"lokad"}


def fail(code, msg):
    print("campaign-%s: %s" % ("ABORT" if code == 2 else "FAIL", msg))
    raise SystemExit(code)


def host_pairs(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


def check_hosts(reps, tag):
    ref = host_pairs(reps[0]["host"])
    for rep in reps[1:]:
        got = host_pairs(rep["host"])
        for k in ref:
            if k in HOST_IGNORE:
                continue
            if got.get(k) != ref[k]:
                fail(2, "%s host mismatch on %s: %r vs %r (%s)" % (tag, k, ref[k], got.get(k), rep["log"]))
    return ref


def check_cross(l0host, l1host):
    for k in l0host:
        if k in HOST_IGNORE:
            continue
        if l1host.get(k) != l0host[k]:
            fail(2, "L0/L1 host mismatch on %s: %r vs %r" % (k, l0host[k], l1host.get(k)))


def med(xs):
    return statistics.median(sorted(xs))


def steady(name, series):
    s = sorted(series)
    m = med(s)
    if m == 0:
        return "zero median"
    if (s[-1] - s[0]) / m > SPREAD_CAP:
        return "spread %.3f over %.2f" % ((s[-1] - s[0]) / m, SPREAD_CAP)
    n = len(s)
    fh = med(s[: n // 2])
    sh = med(s[n // 2 :])
    if abs(fh - sh) / m > HALF_CAP:
        return "half-split %.3f over %.2f" % (abs(fh - sh) / m, HALF_CAP)
    return ""


def main():
    args = sys.argv[1:]
    if "--l1" not in args:
        print(__doc__)
        raise SystemExit(4)
    cut = args.index("--l1")
    l0logs, l1logs = args[:cut], args[cut + 1 :]
    if len(l0logs) < MIN_REPS or len(l1logs) < MIN_REPS:
        fail(4, "need >=%d reps per library, got L0=%d L1=%d" % (MIN_REPS, len(l0logs), len(l1logs)))
    l0 = [base.parse_rep(p) for p in l0logs]
    l1 = [base.parse_rep(p) for p in l1logs]
    h0 = check_hosts(l0, "L0")
    h1 = check_hosts(l1, "L1")
    check_cross(h0, h1)
    for rep in l0 + l1:
        for name in PRIMARY:
            if rep["status"].get(name) != "ok" or name not in rep["cases"]:
                fail(2, "case missing or not ok: %s in %s" % (name, rep["log"]))
    unsteady = []
    for lib, reps in (("L0", l0), ("L1", l1)):
        for rep in reps:
            for name in PRIMARY:
                for eng in ("lok", "ort"):
                    xs = rep["cases"][name]["raw"][eng]
                    if len(xs) < MIN_SAMPLES:
                        fail(4, "fewer than %d samples: %s %s %s has %d" % (MIN_SAMPLES, lib, name, eng, len(xs)))
                    why = steady(name, xs)
                    if why:
                        unsteady.append("%s %s %s %s" % (lib, rep["log"], name, why))
    if unsteady:
        print("campaign-INCONCLUSIVE: unsteady series, no score")
        for u in unsteady:
            print("  " + u)
        print(json.dumps({"verdict": "INCONCLUSIVE", "unsteady": unsteady}, indent=1))
        raise SystemExit(3)
    table = []
    reg = []
    for name in PRIMARY:
        l0m = [med(rep["cases"][name]["raw"]["lok"]) for rep in l0]
        l1m = [med(rep["cases"][name]["raw"]["lok"]) for rep in l1]
        om = [med(rep["cases"][name]["raw"]["ort"]) for rep in l0 + l1]
        L0, L1, O = med(l0m), med(l1m), med(om)
        r0, r1 = L0 / O, L1 / O
        gap = (r0 - r1) / (r0 - 1) if r0 > 1 else None
        respread = (max(l1m) - min(l1m)) / L1 if L1 else 0
        thresh = max(0.03, 2 * respread)
        if L1 / L0 - 1 > thresh:
            reg.append("%s L1/L0=%.4f over %.4f" % (name, L1 / L0, 1 + thresh))
        table.append({"case": name, "L0": L0, "L1": L1, "O": O,
                      "improvement": 1 - L1 / L0, "R0": r0, "R1": r1, "gap_closure": gap})
    r = {row["case"]: row["L1"] / row["L0"] for row in table}
    e5 = (r["e5-8tok"] * r["e5-30tok"]) ** 0.5
    score = 1 - (e5 * r["dinov3-224"] * r["resnet50-224"] * r["gpt2-4tok"]) ** 0.25
    t = {row["case"]: row["improvement"] for row in table}
    targets = {"family": score >= 0.20, "e5-8tok": t["e5-8tok"] >= 0.25, "resnet50-224": t["resnet50-224"] >= 0.25}
    verdict = "PASS" if all(targets.values()) and not reg else ("REGRESSION" if reg else "MISS")
    print("| Case | L0 | L1 | O | 1-L1/L0 | R0 | R1 | gap |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in table:
        g = "n/a" if row["gap_closure"] is None else "%.2f" % row["gap_closure"]
        print("| %s | %.2f | %.2f | %.2f | %.3f | %.2fx | %.2fx | %s |"
              % (row["case"], row["L0"], row["L1"], row["O"],
                 row["improvement"], row["R0"], row["R1"], g))
    print("family-score=%.4f targets=%s verdict=%s" % (score, json.dumps(targets), verdict))
    print(json.dumps({"verdict": verdict, "family_score": score,
                      "targets": targets, "regressions": reg, "table": table}, indent=1))


if __name__ == "__main__":
    main()