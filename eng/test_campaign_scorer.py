"""Golden-log self-test for eng/score_campaign.py (E01/E02 M1).

Run: python eng/test_campaign_scorer.py
Exit 0 when every golden campaign behaves: steady scores PASS, regression
flags REGRESSION, unsteady refuses with INCONCLUSIVE, host drift aborts, and
short series fail on sample count. Standard library only.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORER = os.path.join(ROOT, "eng", "score_campaign.py")
CASES = ["e5-8tok", "e5-30tok", "dinov3-224", "resnet50-224", "gpt2-4tok"]
HOST = ("host=box cpu=x procs=1 affinity=0x10 isa=AVX2 runtime=.NET runt "
        "lokad={v} ort=1.23.2 mode=auto threads=1 rows=canonical iters={n} warmup=3")


def flat(v, n=33):
    return [float(v)] * n


def sloped(v0, v1, n=33):
    return [float(v0)] * (n // 2) + [float(v1)] * (n - n // 2)


def rep(path, lok, ort, lokad="9.9", n=33, isa="AVX2", host_n=None):
    with open(path, "w", encoding="utf-8") as f:
        hn = n if host_n is None else host_n
        f.write(HOST.format(v=lokad, n=hn).replace("isa=AVX2", "isa=" + isa) + "\n")
        for c in CASES:
            lv = lok(c) if callable(lok) else lok
            ov = ort(c) if callable(ort) else ort
            f.write("%s [hdr]: summary line\n" % c)
            f.write("raw lok=[%s] raw ctx=[%s] raw ort=[%s]\n"
                    % (",".join(map(str, lv)), ",".join(map(str, lv)), ",".join(map(str, ov))))
            f.write("case-status %s=ok\n" % c)


def run(l0logs, l1logs):
    p = subprocess.run([sys.executable, SCORER] + l0logs + ["--l1"] + l1logs,
                       capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout + p.stderr


def verdict_of(out):
    start = out.rfind("\n{")
    assert start >= 0, out
    return json.loads(out[start + 1:])


def main():
    with tempfile.TemporaryDirectory() as d:
        mk = lambda n: os.path.join(d, n)
        # Steady winning campaign: L0 lok 20, L1 lok 15, ort 10 everywhere.
        l0 = [mk("l0r%d.log" % i) for i in (1, 2, 3)]
        l1 = [mk("l1r%d.log" % i) for i in (1, 2, 3)]
        for p in l0:
            rep(p, flat(20), flat(10), lokad="0.2.0")
        for p in l1:
            rep(p, flat(15), flat(10), lokad="0.2.1")
        ec, out = run(l0, l1)
        assert ec == 0, out
        v = verdict_of(out)
        assert v["verdict"] == "PASS", out
        assert abs(v["family_score"] - 0.25) < 1e-9, out
        row = [r for r in v["table"] if r["case"] == "e5-8tok"][0]
        assert abs(row["L0"] - 20) < 1e-9 and abs(row["L1"] - 15) < 1e-9, out
        assert abs(row["R0"] - 2.0) < 1e-9 and abs(row["gap_closure"] - 0.5) < 1e-9, out
        # Regression: resnet L1 slower than L0 beyond the band.
        l1b = [mk("l1br%d.log" % i) for i in (1, 2, 3)]
        for p in l1b:
            rep(p, lambda c: flat(30) if c == "resnet50-224" else flat(15),
                flat(10), lokad="0.2.1")
        ec, out = run(l0, l1b)
        assert ec == 0, out
        assert verdict_of(out)["verdict"] == "REGRESSION", out
        # Unsteady: sloped L1 e5-8tok trips the half-split gate.
        l1c = [mk("l1cr%d.log" % i) for i in (1, 2, 3)]
        for p in l1c:
            rep(p, lambda c: sloped(20, 22.5) if c == "e5-8tok" else flat(15),
                flat(10), lokad="0.2.1")
        ec, out = run(l0, l1c)
        assert ec == 3 and "INCONCLUSIVE" in out, out
        # Host drift: one rep with a different ISA aborts.
        l0d = [mk("l0dr%d.log" % i) for i in (1, 2, 3)]
        for i, p in enumerate(l0d):
            rep(p, flat(20), flat(10), lokad="0.2.0", isa="AVX2" if i else "SSE")
        ec, out = run(l0d, l1)
        assert ec == 2 and "ABORT" in out, out
        # Short series fail on sample count, not on steadiness.
        l0e = [mk("l0er%d.log" % i) for i in (1, 2, 3)]
        for p in l0e:
            rep(p, flat(20, 9), flat(10, 9), lokad="0.2.0", n=9, host_n=33)
        ec, out = run(l0e, l1)
        assert ec == 4 and "33" in out, out
    print("campaign scorer self-test: all golden campaigns behave")


if __name__ == "__main__":
    main()