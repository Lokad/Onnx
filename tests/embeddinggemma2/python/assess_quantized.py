"""Assess the pinned weight-only 8-bit export against frozen FP32 references.

This is a precision assessment, not a managed-runtime conformance or speed claim.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform

import numpy as np
import onnx
import onnxruntime as ort

from common import pooled, save_array, session, sha256, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = verify_assets(args.model_dir, "quantized")
    source = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    pins = json.loads((Path(__file__).resolve().parents[1] / "assets.json").read_text(encoding="utf-8"))["onnx"]
    assert source["modelSha256"] == pins["files"]["onnx/model.onnx"]
    assert source["weightsSha256"] == pins["files"]["onnx/model.onnx_data"]
    assert source["tokenizerSha256"] == pins["files"]["tokenizer.json"]
    assert source["casesSha256"] == sha256(args.cases)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    assert [c["id"] for c in source["cases"]] == [c["id"] for c in cases] + ["mixed-batch"]
    path = args.model_dir / "onnx/model_quantized.onnx"
    graph = onnx.load(path, load_external_data=False)
    ops = Counter((n.domain or "ai.onnx") + "::" + n.op_type for n in graph.graph.node)
    layouts = Counter()
    for node in graph.graph.node:
        if node.op_type in ("MatMulNBits", "GatherBlockQuantized"):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            layouts[(node.op_type, attrs.get("bits"), attrs.get("block_size"))] += 1
    model = session(path)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "8-bit weight-only precision assessment; no managed support or performance claim",
              "python": platform.python_version(), "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__,
              "providers": model.get_providers(), "repository": assets["repository"], "revision": assets["revision"],
              "files": {n: {"sha256": sha256(args.model_dir / n), "bytes": (args.model_dir / n).stat().st_size} for n in assets["files"]},
              "referenceSha256": sha256(args.reference / "manifest.json"), "producerSha256": sha256(Path(__file__)),
              "operators": dict(sorted(ops.items())), "quantizationLayouts": [dict(operator=o, bits=b, blockSize=s, count=c) for (o, b, s), c in sorted(layouts.items())],
              "cases": []}
    vectors = {}
    for case in source["cases"]:
        name = case["id"]

        def read(key):
            info = case[key]
            array_path = args.reference / name / info["file"]
            assert sha256(array_path) == info["sha256"], f"Changed reference {name}/{key}"
            array = np.load(array_path, allow_pickle=False)
            assert list(array.shape) == info["shape"] and str(array.dtype) == info["dtype"]
            return array

        ids, mask = read("input_ids"), read("attention_mask")
        feeds = {"input_ids": ids, "attention_mask": mask,
                 **{key: np.empty((0, 512), np.float32) for key in ("image_features", "video_features", "audio_features")}}
        hidden, vector = model.run(["last_hidden_state", "sentence_embedding"], feeds)
        repeated = model.run(["sentence_embedding"], feeds)[0]
        assert vector.shape == (ids.shape[0], 768) and vector.dtype == np.float32 and np.isfinite(vector).all()
        assert vector.tobytes() == repeated.tobytes(), f"Quantized ORT repeatability failed: {name}"
        np.testing.assert_allclose(pooled(hidden, mask), vector, rtol=1e-5, atol=1e-6)
        vectors[name] = vector
        directory = args.output / name
        directory.mkdir()
        record = {"id": name, "byteRepeatable": True, "dimensions": {}}
        for dimension in (128, 256, 512, 768):
            actual = pooled(hidden, mask, dimension)
            reference = read(f"embedding_{dimension}")
            a, r = actual.astype(np.float64), reference.astype(np.float64)
            error = np.abs(a - r)
            cosine = np.sum(a * r, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1))
            relative = np.linalg.norm(a - r, axis=1) / np.linalg.norm(r, axis=1)
            ratio = float(np.max(error / (1e-6 + 1e-5 * np.abs(r))))
            norms = np.linalg.norm(a, axis=1)
            assert np.isfinite(a).all() and np.all(np.abs(norms - 1) <= 1e-5)
            record["dimensions"][str(dimension)] = dict(minimumCosine=float(cosine.min()), maximumRelativeL2=float(relative.max()),
                maxAbsoluteError=float(error.max()), maxComponentAllowanceRatio=ratio,
                meetsFP32Gate=bool(ratio <= 1 and np.all(relative <= 1e-4) and np.all(cosine >= .99999)),
                output=save_array(directory, f"embedding_{dimension}", actual))
        report["cases"].append(record)
        print(f"PASS finite/repeatable quantized CPU output: {name}; cosine768={record['dimensions']['768']['minimumCosine']:.8f}", flush=True)
    query = vectors["query"][0]
    scores = {name: float(query @ vectors[name][0]) for name in ("related", "unrelated", "french")}
    report["semanticScores"] = scores
    report["semanticSmokePassed"] = scores["related"] > scores["unrelated"] and scores["french"] > scores["unrelated"]
    assert report["semanticSmokePassed"], "Quantized semantic smoke failed"
    report["worstCosineByDimensions"] = {str(d): min(c["dimensions"][str(d)]["minimumCosine"] for c in report["cases"]) for d in (128, 256, 512, 768)}
    (args.output / "assessment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS 8-bit precision assessment (separate from FP32 equivalence): " + json.dumps(report["worstCosineByDimensions"]), flush=True)


if __name__ == "__main__":
    main()
