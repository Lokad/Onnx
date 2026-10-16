"""Cross-check the export against Google's CPU FP32 Transformers model."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

from common import formatted_text, numerical_check, save_array, verify_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_assets(args.source_dir, "source")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = AutoConfig.from_pretrained(args.source_dir, local_files_only=True)
    config.vision_config = None
    config.audio_config = None
    tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.source_dir, config=config, dtype=torch.float32,
                                      attn_implementation="eager", local_files_only=True).eval()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    native = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"schemaVersion": 1, "python": platform.python_version(), "torch": torch.__version__,
              "transformers": transformers.__version__, "cases": [], "failures": []}
    by_id = {case["id"]: case for case in cases}
    for record in native["cases"]:
        name = record["id"]
        origin = args.reference / name
        destination = args.output / name
        destination.mkdir(exist_ok=True)
        texts = [formatted_text(case) for case in cases[:3]] if name == "mixed-batch" else formatted_text(by_id[name])
        encoded = tokenizer(texts, add_special_tokens=True, padding=True, return_tensors="np")
        ids = np.load(origin / record["input_ids"]["file"], allow_pickle=False)
        mask = np.load(origin / record["attention_mask"]["file"], allow_pickle=False)
        for key, expected in (("input_ids", ids), ("attention_mask", mask)):
            if encoded[key].dtype != expected.dtype or not np.array_equal(encoded[key], expected):
                raise AssertionError(f"Official/export tokenizer disagreement: {name} {key}")
        with torch.inference_mode():
            output = model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask), return_dict=True)
            hidden = output.last_hidden_state
            weights = torch.from_numpy(mask).unsqueeze(-1).to(torch.float32)
            mean = torch.where(weights != 0, hidden, 0.0).sum(1) / weights.sum(1).clamp_min(1e-9)
            vector = torch.nn.functional.normalize(mean, p=2, dim=-1).numpy().copy()
            hidden_array = hidden.numpy().copy()
        item = {"id": name, "last_hidden_state": save_array(destination, "last_hidden_state", hidden_array),
                "sentence_embedding": save_array(destination, "sentence_embedding", vector)}
        for key, actual, strict in (("last_hidden_state", hidden_array, False), ("sentence_embedding", vector, True)):
            reference = np.load(origin / record[key]["file"], allow_pickle=False)
            try:
                item[key + "Check"] = numerical_check(actual, reference, components=strict)
            except AssertionError as error:
                result["failures"].append({"id": name, "tensor": key, "error": str(error)})
        result["cases"].append(item)
        print(f"CHECK source/export {name}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["failures"]:
        raise AssertionError(f"Source/export disagreement: {result['failures']}")
    print("PASS source model/export/tokenizer agreement", flush=True)


if __name__ == "__main__":
    main()
