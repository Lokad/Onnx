"""Freeze exact benchmark text/tensors and independent owned FP32 vector oracles."""
import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from common import numerical_check, pooled, session, sha256, verify_assets


def text_for_tokens(tokenizer, count):
    sentence = "The telescope records the changing position of Mars above the horizon. "
    prefix = "title: Observations | text: "
    text = prefix
    while len(tokenizer.encode(text + sentence).ids) <= count - 3:
        text += sentence
    for extra in range(count):
        candidate = text + "x " * extra
        if len(tokenizer.encode(candidate).ids) == count:
            return candidate
    raise ValueError(f"Cannot construct exactly {count} tokens")


def write_array(directory, name, values):
    path = directory / name
    with path.open("xb") as stream:
        stream.write(np.ascontiguousarray(values).tobytes())
    return {"file": name, "sha256": sha256(path), "dtype": str(values.dtype), "shape": list(values.shape)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = verify_assets(args.model_dir)
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    model = session(args.model_dir / "onnx/model.onnx")
    args.output.mkdir(parents=True, exist_ok=False)
    preparation = args.model_dir / "prepared-fp32/preparation.json"
    prepared = json.loads(preparation.read_text(encoding="utf-8"))
    graph = args.model_dir / "prepared-fp32/model.onnx"
    assert prepared["graphSha256"] == sha256(graph)
    manifest = {"schemaVersion": 1, "protocol": "embeddinggemma2-owned-vector-v1", "cases": [],
                "modelSha256": assets["files"]["onnx/model.onnx"], "weightsSha256": assets["files"]["onnx/model.onnx_data"],
                "tokenizerSha256": assets["files"]["tokenizer.json"], "preparedGraphSha256": sha256(graph),
                "preparationSha256": sha256(preparation), "producerSha256": sha256(Path(__file__))}
    for name, width, real in (("gemma2-32tok", 32, 32), ("gemma2-128tok", 128, 128),
                              ("gemma2-512tok", 512, 512), ("gemma2-32pad128", 128, 32)):
        text = text_for_tokens(tokenizer, real)
        encoded = tokenizer.encode(text).ids
        ids, mask = np.zeros((1, width), dtype=np.int64), np.zeros((1, width), dtype=np.int64)
        ids[0, :real], mask[0, :real] = encoded, 1
        feeds = {"input_ids": ids, "attention_mask": mask,
                 **{key: np.empty((0, 512), np.float32) for key in ("image_features", "video_features", "audio_features")}}
        hidden, sentence = model.run(["last_hidden_state", "sentence_embedding"], feeds)
        numerical_check(pooled(hidden, mask), sentence, components=True)
        directory = args.output / name
        directory.mkdir()
        record = {"id": name, "sequenceLength": width, "unmaskedTokens": real, "text": text,
                  "input_ids": write_array(directory, "input_ids.i64", ids),
                  "attention_mask": write_array(directory, "attention_mask.i64", mask)}
        for dimension in (128, 256, 512, 768):
            record[f"embedding_{dimension}"] = write_array(directory, f"embedding_{dimension}.f32", pooled(hidden, mask, dimension))
        manifest["cases"].append(record)
        print(f"PASS frozen benchmark {name}: {real} real / {width} padded tokens", flush=True)
    (args.output / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
