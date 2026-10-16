#!/usr/bin/env python3
"""Prepare the pinned EmbeddingGemma 2 export for managed text execution."""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/embeddinggemma2/python"))
from common import numerical_check, pooled, session, sha256, verify_assets


class Expander:
    def __init__(self, model):
        self.initializers = list(model.graph.initializer)
        self.constants = {t.name: t for t in self.initializers}
        self.info = {v.name: v for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
        self.nodes = []
        self.serial = 0
        self.counts = collections.Counter()

    def name(self, purpose):
        self.serial += 1
        return f"__lokad_eg2/{self.serial}/{purpose}"

    def const(self, values, dtype):
        name = self.name("constant")
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def emit(self, op, inputs, output=None, **attributes):
        output = output or self.name(op)
        self.nodes.append(helper.make_node(op, inputs, [output], name=self.name(op + "_node"), **attributes))
        return output

    def rms(self, x, scale, output, epsilon):
        axes = self.const([-1], np.int64)
        squared = self.emit("Mul", [x, x])
        mean = self.emit("ReduceMean", [squared, axes], keepdims=1)
        variance = self.emit("Add", [mean, self.const(epsilon, np.float32)])
        inverse = self.emit("Pow", [variance, self.const(-0.5, np.float32)])
        normalized = self.emit("Mul", [x, inverse])
        self.emit("Mul", [normalized, scale], output)

    def rotary(self, node, attrs):
        if attrs.get("interleaved", 0) != 0:
            raise ValueError("Only the pinned noninterleaved rotary layout is supported")
        x, position, cos, sin = node.input
        if position not in self.constants or np.any(numpy_helper.to_array(self.constants[position]) != 0):
            raise ValueError("The pinned rotary position offset must be constant zero")
        x_shape = self.info[x].type.tensor_type.shape.dim
        cos_shape = self.info[cos].type.tensor_type.shape.dim
        if len(x_shape) != 3 or len(cos_shape) != 2:
            raise ValueError("Unexpected rotary rank")
        half = cos_shape[-1].dim_value
        if half not in (128, 256):
            raise ValueError(f"Unexpected rotary cache width: {half}")
        dim = half * 2
        shaped = self.emit("Reshape", [x, self.const([0, 0, -1, dim], np.int64)])
        first, second = self.name("first"), self.name("second")
        self.nodes.append(helper.make_node("Split", [shaped, self.const([half, half], np.int64)],
                                           [first, second], name=self.name("split"), axis=-1))
        neg = self.emit("Neg", [second])
        rotated = self.emit("Concat", [neg, first], axis=-1)
        axes = self.const([0, 2], np.int64)
        cosine = self.emit("Unsqueeze", [self.emit("Concat", [cos, cos], axis=-1), axes])
        sine = self.emit("Unsqueeze", [self.emit("Concat", [sin, sin], axis=-1), axes])
        added = self.emit("Add", [self.emit("Mul", [shaped, cosine]), self.emit("Mul", [rotated, sine])])
        self.emit("Reshape", [added, self.const([0, 0, -1], np.int64)], node.output[0])

    def expand(self, node):
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "SimplifiedLayerNormalization" and node.domain in ("", "ai.onnx"):
            if attrs.get("axis", -1) != -1 or attrs.get("stash_type", 1) != 1 or len(node.output) != 1:
                raise ValueError("Unexpected simplified normalization contract")
            self.rms(node.input[0], node.input[1], node.output[0], attrs.get("epsilon", 1e-5))
        elif node.op_type == "SkipSimplifiedLayerNormalization" and node.domain == "com.microsoft":
            if any(node.output[i] for i in (1, 2) if i < len(node.output)):
                raise ValueError("Unexpected optional skip-normalization output")
            sum_output = node.output[3] if len(node.output) > 3 and node.output[3] else self.name("residual")
            has_bias = len(node.input) > 3 and bool(node.input[3])
            summed = self.emit("Add", list(node.input[:2]), None if has_bias else sum_output)
            if has_bias:
                self.emit("Add", [summed, node.input[3]], sum_output)
            self.rms(sum_output, node.input[2], node.output[0], attrs.get("epsilon", 1e-5))
        elif node.op_type == "RotaryEmbedding" and node.domain == "com.microsoft":
            self.rotary(node, attrs)
        else:
            self.nodes.append(node)
            return
        self.counts[(node.domain + "::" if node.domain else "") + node.op_type] += 1


def rewrite_model(original):
    model = copy.deepcopy(original)
    expander = Expander(model)
    names = {o for n in model.graph.node for o in n.output}
    merge = "/model/multimodal_merge/Reshape_embeds/output_0"
    text = "/model/embed_tokens/Gather/output_0"
    if merge in names and text not in names:
        raise ValueError("Missing original text embedding producer")
    for node in model.graph.node:
        node.input[:] = [text if value == merge else value for value in node.input]
        expander.expand(node)
    needed = {v.name for v in model.graph.output}
    kept = []
    for node in reversed(expander.nodes):
        if any(value in needed for value in node.output if value):
            kept.append(node)
            needed.update(value for value in node.input if value)
    model.graph.ClearField("node")
    model.graph.node.extend(reversed(kept))
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(t for t in expander.initializers if t.name in needed)
    inputs = [v for v in model.graph.input if v.name in needed]
    model.graph.ClearField("input")
    model.graph.input.extend(inputs)
    infos = [v for v in model.graph.value_info if v.name in needed]
    model.graph.ClearField("value_info")
    model.graph.value_info.extend(infos)
    if any(n.domain not in ("", "ai.onnx") for n in model.graph.node):
        raise ValueError("Unexpected remaining custom operator")
    imports = [v for v in model.opset_import if v.domain in ("", "ai.onnx")]
    model.ClearField("opset_import")
    model.opset_import.extend(imports)
    return model, dict(expander.counts)


def verify_prepared(model_path: Path, reference_path: Path) -> dict:
    reference = json.loads((reference_path / "manifest.json").read_text(encoding="utf-8"))
    model = session(model_path)
    results = []
    for record in reference["cases"]:
        origin = reference_path / record["id"]
        def load(name):
            info = record[name]
            path = origin / info["file"]
            if sha256(path) != info["sha256"]:
                raise ValueError(f"Reference artifact changed: {path}")
            return np.load(path, allow_pickle=False)
        ids, mask = load("input_ids"), load("attention_mask")
        hidden = model.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]
        result = {"id": record["id"], "hidden": numerical_check(hidden, load("last_hidden_state"), components=False)}
        for dimension in (128, 256, 512, 768):
            result[str(dimension)] = numerical_check(pooled(hidden, mask, dimension), load(f"embedding_{dimension}"), components=True)
        results.append(result)
        print(f"PASS prepared native parity {record['id']}", flush=True)
    return {"cases": results, "referenceManifestSha256": sha256(reference_path / "manifest.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--reference", type=Path, default=Path("artifacts/embeddinggemma2/reference"))
    args = parser.parse_args()
    if args.output.resolve() in (args.source.resolve(), (args.source / "onnx").resolve()):
        raise ValueError("Prepared output must be separate from original asset directories")
    assets = verify_assets(args.source)
    original_path = args.source / "onnx/model.onnx"
    model = onnx.load(original_path, load_external_data=False)
    output_info = next(v for v in model.graph.output if v.name == "last_hidden_state")
    model.graph.ClearField("output")
    model.graph.output.append(output_info)
    model, changes = rewrite_model(model)
    if [v.name for v in model.graph.input] != ["input_ids", "attention_mask"]:
        raise ValueError("Prepared graph must have exactly two text inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    locations = {e.value for t in model.graph.initializer for e in t.external_data if e.key == "location"}
    for location in locations:
        if Path(location).name != location:
            raise ValueError("Unexpected external weight location")
        source = original_path.parent / location
        target = args.output / location
        if target.exists():
            if os.path.samefile(source, target):
                # ONNX 1.22's external-data validator rejects multiple links.
                # Unlink only this exact prepared path, preserving the source.
                target.unlink()
                shutil.copyfile(source, target)
            elif sha256(source) != sha256(target):
                raise ValueError(f"Mismatched prepared sidecar: {target}")
        else:
            shutil.copyfile(source, target)
    path = args.output / "model.onnx"
    onnx.save_model(model, path)
    onnx.checker.check_model(str(path))
    operators = collections.Counter(n.op_type for n in model.graph.node)
    tokenizer_path = args.output / "tokenizer.json"
    if tokenizer_path.exists() and sha256(tokenizer_path) != assets["files"]["tokenizer.json"]:
        raise ValueError("Mismatched prepared tokenizer")
    if not tokenizer_path.exists():
        shutil.copyfile(args.source / "tokenizer.json", tokenizer_path)
    receipt = {"schemaVersion": 1, "sourceGraphSha256": sha256(original_path), "graphSha256": sha256(path),
               "tokenizerSha256": sha256(tokenizer_path), "weightsSha256": assets["files"]["onnx/model.onnx_data"],
               "preparerSha256": sha256(Path(__file__)), "rewrites": changes, "operators": dict(operators),
               "inputs": [v.name for v in model.graph.input], "output": "last_hidden_state", "precision": "fp32"}
    if args.verify:
        receipt["verification"] = verify_prepared(path, args.reference)
    (args.output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"PASS text-only rewrite: {len(model.graph.node)} nodes; {sha256(path)}", flush=True)


if __name__ == "__main__":
    main()
