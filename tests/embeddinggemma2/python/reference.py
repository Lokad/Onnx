"""Generate lossless tokenizer/model references without loading managed code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import onnxruntime as ort
import tokenizers

from common import formatted_text, numerical_check, pooled, save_array, session, sha256, verify_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-only", action="store_true")
    args = parser.parse_args()
    tokenizer_path = args.model_dir / "tokenizer.json"
    tok = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    model = None
    if not args.tokens_only:
        verify_assets(args.model_dir)
        model = session(args.model_dir / "onnx/model.onnx")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schemaVersion": 1, "python": platform.python_version(), "numpy": np.__version__,
                "onnxruntime": ort.__version__, "tokenizers": tokenizers.__version__,
                "tokenizerSha256": sha256(tokenizer_path), "casesSha256": sha256(args.cases), "cases": []}
    manifest["producerSha256"] = sha256(Path(__file__))
    if model is not None:
        manifest["modelSha256"] = sha256(args.model_dir / "onnx/model.onnx")
        manifest["weightsSha256"] = sha256(args.model_dir / "onnx/model.onnx_data")
    embeddings = {}
    entries = []
    for case in cases:
        encoded = tok.encode(formatted_text(case), add_special_tokens=True)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.ones_like(ids)
        if ids.shape[1] > 512:
            raise ValueError(f"Case {case['id']} exceeds 512 tokens: {ids.shape[1]}")
        entries.append((case["id"], ids, mask))
    # Exercise right padding independently of any model-specific Python wrapper.
    max_length = max(ids.shape[1] for _, ids, _ in entries[:3])
    batch_ids = np.zeros((3, max_length), dtype=np.int64)
    batch_mask = np.zeros_like(batch_ids)
    for index, (_, ids, _) in enumerate(entries[:3]):
        batch_ids[index, :ids.shape[1]] = ids[0]
        batch_mask[index, :ids.shape[1]] = 1
    entries.append(("mixed-batch", batch_ids, batch_mask))
    for name, ids, mask in entries:
        destination = args.output / name
        destination.mkdir(exist_ok=True)
        record = {"id": name, "input_ids": save_array(destination, "input_ids", ids),
                  "attention_mask": save_array(destination, "attention_mask", mask)}
        if model is not None:
            feeds = {"input_ids": ids, "attention_mask": mask,
                     **{key: np.empty((0, 512), np.float32) for key in
                        ("image_features", "video_features", "audio_features")}}
            hidden, vector = model.run(["last_hidden_state", "sentence_embedding"], feeds)
            repeated = model.run(["sentence_embedding"], feeds)[0]
            if vector.tobytes() != repeated.tobytes():
                raise AssertionError(f"ORT repeatability failed: {name}")
            record["poolingCheck"] = numerical_check(pooled(hidden, mask), vector, components=True)
            record["last_hidden_state"] = save_array(destination, "last_hidden_state", hidden)
            record["sentence_embedding"] = save_array(destination, "sentence_embedding", vector)
            for dimension in (128, 256, 512, 768):
                result = pooled(hidden, mask, dimension)
                record[f"embedding_{dimension}"] = save_array(destination, f"embedding_{dimension}", result)
            embeddings[name] = vector
        manifest["cases"].append(record)
        print(f"PASS reference {name}: {list(ids.shape)}", flush=True)
    if model is not None:
        query = embeddings["query"][0]
        scores = {name: float(query @ embeddings[name][0]) for name in ("related", "unrelated", "french")}
        if scores["related"] <= scores["unrelated"] or scores["french"] <= scores["unrelated"]:
            raise AssertionError(f"Semantic smoke failed: {scores}")
        manifest["semanticScores"] = scores
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("PASS tokenizer reference" if model is None else "PASS native FP32 reference", flush=True)


if __name__ == "__main__":
    main()
