"""Check integrated Data API IDs/masks/vectors using independent frozen arrays."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import numerical_check, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    native = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    actual = json.loads((args.actual / "manifest.json").read_text(encoding="utf-8"))
    assert native["casesSha256"] == actual["casesSha256"] == sha256(args.cases)
    assert native["tokenizerSha256"] == actual["tokenizerSha256"]
    assert actual["referenceSha256"] == sha256(args.reference / "manifest.json")
    assert actual["cases"] == [{"id": c["id"], "api": c["role"] != "raw"} for c in cases], "API case coverage differs"
    assert len(set(actual["spaceIds"].values())) == 4
    by_id = {c["id"]: c for c in native["cases"]}
    reads = set()

    def managed(relative):
        path = args.actual / relative
        assert sha256(path) == actual["outputs"][relative], "Changed managed output"
        reads.add(relative)
        return np.load(path, allow_pickle=False)

    def oracle(name, tensor):
        info = by_id[name][tensor]
        path = args.reference / name / info["file"]
        assert sha256(path) == info["sha256"], "Changed native oracle"
        return np.load(path, allow_pickle=False)

    checks = []
    for case in cases:
        name = case["id"]
        for key in ("input_ids", "attention_mask"):
            value, expected = managed(f"{name}/{key}.npy"), oracle(name, key)
            assert value.dtype == expected.dtype and np.array_equal(value, expected), f"Exact integrated inputs differ: {name} {key}"
        if case["role"] != "raw":
            for dimension in (128, 256, 512, 768):
                key = f"embedding_{dimension}"
                for prefix in (["", "batched/"] if case["role"] == "document" else [""]):
                    vector = managed(f"{prefix}{name}/{key}.npy")
                    check = numerical_check(vector, oracle(name, key), components=True)
                    assert np.all(np.abs(np.linalg.norm(vector.astype(np.float64), axis=-1) - 1) <= 1e-5)
                    checks.append({"id": name, "dimensions": dimension, "batch": bool(prefix), **check})
    covered = []
    for batch in actual["batches"]:
        rows = [oracle(name, "input_ids")[0] for name in batch["ids"]]
        assert batch["tokenCounts"] == [len(row) for row in rows]
        ids = np.zeros((len(rows), max(map(len, rows))), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = row
            mask[i, :len(row)] = 1
        for key, expected in (("input_ids", ids), ("attention_mask", mask)):
            value = managed(batch["directory"] + "/" + key + ".npy")
            assert value.dtype == expected.dtype and np.array_equal(value, expected), "Integrated batch padding differs"
        covered.extend(batch["ids"])
    assert covered == [c["id"] for c in cases if c["role"] == "document"]
    assert reads == set(actual["outputs"]), "Unverified or missing API outputs"
    (args.actual / "conformance.json").write_text(json.dumps({"schemaVersion": 1, "checks": checks,
        "manifestSha256": sha256(args.actual / "manifest.json"), "producerSha256": sha256(Path(__file__))}, indent=2) + "\n", encoding="utf-8")
    print(f"PASS exact integrated inputs for {len(cases)} cases; {len(checks)} single/batched vectors at all dimensions")


if __name__ == "__main__":
    main()
