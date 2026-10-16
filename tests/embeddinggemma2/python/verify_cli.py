"""Exercise the public raw-text CLI and compare its owned JSON vectors with ORT."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from common import numerical_check, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    reference = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    assert reference["casesSha256"] == sha256(args.cases), "Changed cases"
    native = {c["id"]: c for c in reference["cases"]}
    query = next(c for c in cases if c["id"] == "query")
    documents = [c for c in cases if c["role"] == "document"]
    jsonl = "".join(json.dumps({"id": c["id"], "text": c["text"] * c.get("repeat", 1),
                               "title": c.get("title")}, ensure_ascii=False) + "\n" for c in documents)
    base = ["dotnet", str(args.cli), "embed", str(args.model_dir)]
    args.output.mkdir(parents=True, exist_ok=True)
    checks, spaces = [], set()
    for dimension in (128, 256, 512, 768):
        query_run = subprocess.run([*base, "--query", query["text"], "--dimensions", str(dimension)],
                                   text=True, encoding="utf-8", capture_output=True, check=True)
        document_run = subprocess.run([*base, "--documents", "-", "--batch-size", "3", "--dimensions", str(dimension)],
                                      input=jsonl, text=True, encoding="utf-8", capture_output=True, check=True)
        query_records = [json.loads(line) for line in query_run.stdout.splitlines()]
        records = [json.loads(line) for line in document_run.stdout.splitlines()]
        assert len(query_records) == 1 and query_records[0]["id"] is None
        assert [r["id"] for r in records] == [c["id"] for c in documents]
        all_records = [("query", query_records[0]), *[(r["id"], r) for r in records]]
        current_spaces = set()
        for name, record in all_records:
            assert record["model"] == "embeddinggemma-2" and record["precision"] == "fp32" and record["dimensions"] == dimension
            info = native[name][f"embedding_{dimension}"]
            path = args.reference / name / info["file"]
            assert sha256(path) == info["sha256"]
            vector = np.asarray(record["embedding"], dtype=np.float32).reshape(1, dimension)
            check = numerical_check(vector, np.load(path, allow_pickle=False), components=True)
            assert abs(float(np.linalg.norm(vector.astype(np.float64))) - 1) <= 1e-5
            mask = np.load(args.reference / name / native[name]["attention_mask"]["file"], allow_pickle=False)
            assert record["tokenCount"] == int(mask.sum())
            current_spaces.add(record["spaceId"])
            checks.append({"id": name, "dimensions": dimension, **check})
        assert len(current_spaces) == 1, "Query and document space identities differ"
        spaces.update(current_spaces)
        (args.output / f"vectors-{dimension}.json").write_text(json.dumps(all_records) + "\n", encoding="utf-8")
        print(f"PASS CLI query and {len(documents)} batched JSONL documents: {dimension} dimensions", flush=True)
    assert len(spaces) == 4, "Dimensions must have distinct space identities"
    rejected_inputs = [(["--query", "test", "--max-tokens", "2"], None),
                       (["--documents", "-"], '{"id":"bad","text":null}\n')]
    boundary = next((c for c in cases if c["id"] == "limit-512"), None)
    if boundary:
        rejected_inputs.append((["--documents", "-"], json.dumps({"id": "overflow-513", "title": boundary["title"],
            "text": boundary["text"] * (boundary["repeat"] + 1)}) + "\n"))
    for extra, input_text in rejected_inputs:
        rejected = subprocess.run([*base, *extra], input=input_text, text=True, encoding="utf-8", capture_output=True)
        assert rejected.returncode == 5 and not rejected.stdout and rejected.stderr, "Invalid input did not fail cleanly"
    (args.output / "conformance.json").write_text(json.dumps({"schemaVersion": 1, "checks": checks,
        "referenceSha256": sha256(args.reference / "manifest.json"), "cliSha256": sha256(args.cli),
        "producerSha256": sha256(Path(__file__))}, indent=2) + "\n", encoding="utf-8")
    print("PASS raw-text CLI conformance and input rejection", flush=True)


if __name__ == "__main__":
    main()
