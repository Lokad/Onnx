"""Verify every managed hidden state and vector against the frozen native oracle."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import numerical_check, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    actual = json.loads((args.actual / "manifest.json").read_text(encoding="utf-8"))
    expected_ids = [c["id"] for c in reference["cases"]]
    if [c["id"] for c in actual["cases"]] != expected_ids:
        raise AssertionError("Managed case coverage/order differs from the oracle")
    results = []
    for case in reference["cases"]:
        name = case["id"]
        record = {"id": name}
        for tensor in ["last_hidden_state", *[f"embedding_{d}" for d in (128, 256, 512, 768)]]:
            native_path = args.reference / name / case[tensor]["file"]
            if sha256(native_path) != case[tensor]["sha256"]:
                raise AssertionError("Changed native reference: " + str(native_path))
            native = np.load(native_path, allow_pickle=False)
            managed = np.load(args.actual / name / (tensor + ".npy"), allow_pickle=False)
            record[tensor] = numerical_check(managed, native, components=tensor != "last_hidden_state")
            if tensor != "last_hidden_state":
                norms = np.linalg.norm(managed.astype(np.float64), axis=-1)
                if np.any(np.abs(norms - 1) > 1e-5):
                    raise AssertionError(f"Managed vector is not unit length: {name} {tensor}")
        results.append(record)
        print(f"PASS managed/native numerical agreement {name}", flush=True)
    (args.actual / "conformance.json").write_text(json.dumps({"schemaVersion": 1, "cases": results}, indent=2) + "\n", encoding="utf-8")
    print("PASS complete managed/native conformance", flush=True)


if __name__ == "__main__":
    main()
