"""Seeded single-op differential conformance: Lokad.Onnx vs frozen ORT references.

Explicit lane, never part of the default `dotnet test` run:
    eng/test-opfuzz.ps1 [-Python <python>]
or directly:
    dotnet build tests/Lokad.Onnx.OpDump -c Release
    python -m pytest tests/opfuzz/python -q
Corpus: tests/opfuzz/corpus/<case>/{model.onnx,in_*.txt,ref_*.txt,meta.json},
frozen by tests/opfuzz/generate/corpus.py (explicit maintenance action only).
"""
import glob
import json
import os
import subprocess

import numpy as np
import pytest

RTOL, ATOL = 1e-5, 1e-6
MODES = ["scalar", "simd", "intrinsics"]

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.normpath(os.path.join(HERE, "..", "corpus"))
RUNNER = os.environ.get(
    "OPDUMP_DLL",
    os.path.normpath(os.path.join(
        HERE, "..", "..", "Lokad.Onnx.OpDump", "bin", "Release", "net10.0", "Lokad.Onnx.OpDump.dll")),
)

DTYPES = {"float32": np.float32, "float64": np.float64, "int64": np.int64, "int32": np.int32, "bool": np.bool_, "uint32": np.uint32, "uint64": np.uint64, "int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16}


def read_txt(path):
    with open(path) as f:
        head = f.readline().split()
        dtype, rank = head[0], int(head[1])
        shape = tuple(int(x) for x in head[2:2 + rank])
        text = f.read().strip()
    if not text:
        return np.empty(shape, dtype=DTYPES[dtype])
    if DTYPES[dtype] in (np.int64, np.int32, np.uint32, np.uint64, np.int8, np.uint8, np.int16, np.uint16):
        # Integer refs must parse exactly: the float64 intermediate rounds
        # values beyond 2**53 (INT64_MAX collapsed to INT64_MIN, passing
        # exact comparison vacuously).
        return np.fromstring(text, dtype=DTYPES[dtype], sep=" ").reshape(shape)
    vals = np.fromstring(text, sep=" ")
    return vals.astype(DTYPES[dtype]).reshape(shape)


def cases():
    return sorted(
        d for d in os.listdir(CORPUS) if os.path.isdir(os.path.join(CORPUS, d))
    )


@pytest.mark.parametrize("case", cases())
def test_case_matches_ort_reference(case, tmp_path):
    if not os.path.isfile(RUNNER):
        pytest.fail("OpDump runner not built: " + RUNNER + " (build tests/Lokad.Onnx.OpDump -c Release first)")
    d = os.path.join(CORPUS, case)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    assert meta["seed"] == 20260904, "corpus was not produced by the pinned generator"
    refs = sorted(glob.glob(os.path.join(d, "ref_*.txt")))
    assert refs, "no frozen references in " + case
    inputs = []
    for ip in sorted(glob.glob(os.path.join(d, "in_*.txt"))):
        inputs += ["--input", os.path.basename(ip)[3:-4] + "=" + ip]
    breaches = []
    for mode in MODES:
        od = os.path.join(str(tmp_path), mode)
        os.makedirs(od)
        r = subprocess.run(
            ["dotnet", RUNNER, "--model", os.path.join(d, "model.onnx"),
             "--mode", mode] + inputs + ["--outdir", od],
            capture_output=True, text=True)
        if r.returncode == 3 and mode == "intrinsics":
            pytest.skip("x86 FMA not available on this machine")
        assert r.returncode == 0, "runner failed for %s/%s: %s" % (case, mode, r.stderr.strip().splitlines()[-1:])
        for rp in refs:
            name = os.path.basename(rp)[4:-4]
            got = glob.glob(os.path.join(od, "dotnet_" + name + ".txt"))
            if not got:
                breaches.append("%s: missing output %s" % (mode, name))
                continue
            ref = read_txt(rp)
            new = read_txt(got[0])
            if ref.shape != new.shape or ref.dtype != new.dtype:
                breaches.append("%s: %s shape/dtype %s/%s vs %s/%s" % (mode, name, ref.shape, ref.dtype, new.shape, new.dtype))
                continue
            if ref.size == 0:
                continue
            if ref.dtype in (np.int64, np.int32, np.uint32, np.uint64, np.int8, np.uint8, np.int16, np.uint16, np.bool_):
                if not np.array_equal(ref, new):
                    breaches.append("%s: %s int mismatch" % (mode, name))
                continue
            denom = np.maximum(np.abs(ref), 1e-12)
            worst_a = float(np.max(np.abs(ref - new)))
            worst_r = float(np.max(np.abs(ref - new) / denom))
            if not np.allclose(new, ref, rtol=RTOL, atol=ATOL, equal_nan=True):
                breaches.append("%s: %s max_abs=%.3e max_rel=%.3e" % (mode, name, worst_a, worst_r))
    divergence = meta.get("known_divergence")
    if breaches and divergence:
        pytest.xfail("known ORT divergence: " + divergence)
    if not breaches and divergence:
        pytest.fail("stale known_divergence flag on %s: case now passes, remove the flag" % case)
    assert not breaches, "case %s breaches:\n%s" % (case, "\n".join(breaches))
