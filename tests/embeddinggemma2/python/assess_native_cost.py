"""One fresh-process native cost diagnostic; never calibrated release evidence.

Run separately for each precision/case, pinned before Python starts with taskset.
Use the existing isolated C# campaigns for any Lokad-versus-ORT claim.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import onnxruntime as ort

from common import numerical_check, session, sha256, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "weight8"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run this cost diagnostic on Linux with launch-time CPU affinity")
    import resource
    assert os.sched_getaffinity(0) == {args.cpu}, "Pin the process before Python starts"
    assert not args.output.exists(), "Choose a fresh diagnostic output"
    pins = verify_assets(args.model_dir, "quantized" if args.precision == "weight8" else "onnx")
    manifest = json.loads(args.fixtures.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "embeddinggemma2-owned-vector-v1"
    fp32 = json.loads((Path(__file__).resolve().parents[1] / "assets.json").read_text(encoding="utf-8"))["onnx"]["files"]
    assert (manifest["modelSha256"], manifest["weightsSha256"], manifest["tokenizerSha256"]) == (
        fp32["onnx/model.onnx"], fp32["onnx/model.onnx_data"], fp32["tokenizer.json"])
    case = next(c for c in manifest["cases"] if c["id"] == args.case)

    def read(key):
        info = case[key]
        path = args.fixtures.parent / args.case / info["file"]
        assert Path(info["file"]).name == info["file"] and sha256(path) == info["sha256"]
        result = np.fromfile(path, dtype=info["dtype"]).reshape(info["shape"])
        assert np.isfinite(result).all()
        return result

    ids, mask, expected = (read(key) for key in ("input_ids", "attention_mask", "embedding_768"))
    feeds = {"input_ids": ids, "attention_mask": mask,
             **{name: np.empty((0, 512), np.float32) for name in ("image_features", "video_features", "audio_features")}}
    filename = "onnx/model_quantized.onnx" if args.precision == "weight8" else "onnx/model.onnx"
    start = time.perf_counter_ns()
    model = session(args.model_dir / filename)
    load_ns = time.perf_counter_ns() - start

    def execute():
        return model.run(["sentence_embedding"], feeds)[0].copy()

    first = execute()
    assert first.shape == (1, 768) and first.dtype == np.float32 and np.isfinite(first).all()
    assert np.max(np.abs(np.linalg.norm(first.astype(np.float64), axis=1) - 1)) <= 1e-5
    if args.precision == "fp32":
        numerical_check(first, expected, components=True)
    a, b = first.astype(np.float64), expected.astype(np.float64)
    cosine = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    warmup_ns = []
    warm_start = time.perf_counter_ns()
    while True:
        start = time.perf_counter_ns()
        value = execute()
        warmup_ns.append(time.perf_counter_ns() - start)
        del value
        steady = len(warmup_ns) >= 32 and sum(warmup_ns) >= 1_000_000_000
        steady = steady and (max(warmup_ns[-9:]) - min(warmup_ns[-9:])) / statistics.median(warmup_ns[-9:]) <= .10
        assert time.perf_counter_ns() - warm_start <= 60_000_000_000, "Warmup exceeded 60 seconds"
        if steady:
            break
        assert len(warmup_ns) < 1000, "Warmup did not converge"
    samples_ns = []
    for _ in range(33):
        start = time.perf_counter_ns()
        value = execute()
        samples_ns.append(time.perf_counter_ns() - start)
        del value
    assert first.tobytes() == execute().tobytes(), "Same-engine byte repeatability failed"
    assert os.sched_getaffinity(0) == {args.cpu}
    report = dict(scope="unscored native precision/cost diagnostic; no A/A or managed speed claim",
                  timing="optimized original ORT sentence_embedding plus owned copy; loading/tokenization outside",
                  precision=args.precision, case=args.case, dimensions=768, cpu=args.cpu,
                  python=platform.python_version(), numpy=np.__version__, onnxruntime=ort.__version__,
                  platform=platform.platform(), providers=model.get_providers(), inference_threads=1,
                  files=pins["files"], revision=pins["revision"], fixture_sha256=sha256(args.fixtures),
                  producer_sha256=sha256(Path(__file__)), load_ns=load_ns, warmup_ns=warmup_ns,
                  samples_ns=samples_ns, median_ms=statistics.median(samples_ns) / 1e6,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  minimum_fp32_cosine=cosine, same_engine_byte_repeatable=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"PASS unscored native {args.precision} {args.case}: {report['median_ms']:.3f} ms, "
          f"peak {report['peak_rss_bytes'] / 2**20:.1f} MiB; cosine {cosine:.8f}", flush=True)


if __name__ == "__main__":
    main()
