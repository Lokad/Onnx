"""Regenerate the single-CPU baseline table from canonical Bench logs.

Usage: python eng/parse_baseline.py <rep1.log> [rep2.log ...]
Reads host, confinement, case-status, case-header, summary, and raw sample
lines; recomputes best, median, p95, and max per engine; verifies every
expected case is ok in every rep; and prints the markdown table plus a JSON
summary to stdout (redirect to a file). Standard library only.
"""
import json
import pathlib
import statistics
import sys

ORDER = ["e5-8tok", "e5-30tok", "dinov3-224", "resnet50-224", "gpt2-4tok"]


def parse_rep(path):
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    host = next((l for l in text if l.startswith("host=")), "")
    conf = next((l for l in text if l.startswith("confinement ")), "")
    status = {}
    headers = {}
    for l in text:
        if l.startswith("case-status "):
            k, _, v = l[len("case-status "):].partition("=")
            status[k] = v
        elif l.startswith("case "):
            k, _, v = l[len("case "):].partition(" ")
            headers[k] = v
    cases = {}
    for i, l in enumerate(text):
        if "raw lok=[" in l:
            name = text[i - 1].split(" [", 1)[0] if i > 0 else "?"
            raws = {}
            for key in ("lok", "ctx", "ort"):
                tag = "raw " + key + "=["
                j = l.find(tag)
                k = l.find("]", j)
                raws[key] = [float(v) for v in l[j + len(tag):k].split(",") if v.strip()]
            cases[name] = {"summary": text[i - 1] if i > 0 else "", "raw": raws}
    return {"log": str(path), "host": host, "confinement": conf,
            "status": status, "headers": headers, "cases": cases}


def stats(xs):
    s = sorted(xs)
    n = len(s)
    p95 = s[min(n - 1, -(-n * 95 // 100) - 1)]
    return {"n": n, "best": s[0], "median": statistics.median(s), "p95": p95, "max": s[-1]}


def main():
    logs = sys.argv[1:]
    if not logs:
        raise SystemExit("usage: python eng/parse_baseline.py <rep1.log> [rep2.log ...]")
    reps = [parse_rep(p) for p in logs]
    for rep in reps:
        for name in ORDER:
            if rep["status"].get(name) != "ok" or name not in rep["cases"]:
                raise SystemExit("case missing or not ok: %s in %s" % (name, rep["log"]))
    table = []
    for name in ORDER:
        rep_cells = []
        for rep in reps:
            lok = stats(rep["cases"][name]["raw"]["lok"])
            ort = stats(rep["cases"][name]["raw"]["ort"])
            rep_cells.append({"lokad": lok, "ort": ort, "ratio": lok["median"] / ort["median"]})
        lm = [c["lokad"]["median"] for c in rep_cells]
        om = [c["ort"]["median"] for c in rep_cells]
        rr = [c["ratio"] for c in rep_cells]
        table.append({"case": name, "reps": rep_cells,
                      "lokad_median_of_medians": statistics.median(lm),
                      "ort_median_of_medians": statistics.median(om),
                      "ratio_span": [min(rr), max(rr)],
                      "lokad_rep_spread": [min(lm), max(lm)],
                      "ort_rep_spread": [min(om), max(om)]})
    head = "| Case | " + " | ".join("rep%d L/ORT ms" % (i + 1) for i in range(len(logs))) + " | Lokad / ORT |"
    bar = "|---|" + "---:|" * len(logs) + "---|"
    print(head)
    print(bar)
    for row in table:
        cells = ["%.1f / %.1f" % (c["lokad"]["median"], c["ort"]["median"]) for c in row["reps"]]
        lo, hi = row["ratio_span"]
        print("| " + row["case"] + " | " + " | ".join(cells) + " | %.1f-%.1fx |" % (lo, hi))
    print("", flush=True)
    print(json.dumps({"reps": reps, "table": table}, indent=1))


if __name__ == "__main__":
    main()
