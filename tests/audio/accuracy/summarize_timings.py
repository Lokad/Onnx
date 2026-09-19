"""Summarize existing September 19 API timings; never run inference or rewrite receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root):
    root = Path(root)
    base = root / "artifacts/asr-labeled-20260919"
    receipt_path = base / "receipt.json"
    assert sha(receipt_path) == "a4855ce825f4d5f1ea931440d031eb3a0aa5b3a0fee1fddbcf4d5b51500754b7"
    receipt = read(receipt_path)

    def verified(relative):
        path = base / relative
        pin = receipt["files"][relative]
        assert sha(path) == pin["sha256"] and path.stat().st_size == pin["bytes"]
        return read(path)

    audio = verified("inputs/audio.json")
    assert len(audio["cases"]) == 20
    records = {}
    for family in ("parakeet", "whisper"):
        result = verified(f"managed-{family}/result.json")
        process = read(base / f"managed-{family}-process.json")
        job = next(j for j in receipt["jobs"] if j["pid"] == process["pid"])
        assert process["complete"] and process["code"] == job["code"] == 0
        assert process["create_time"] == job["create_time"] and process["command"] == job["command"]
        assert process["peak_rss"] == job["peak_rss"] and process["affinity"] == [2]
        assert all(m["affinity"] == [2] for sample in process["samples"] for m in sample["members"])
        assert result["passed"] and result["runtime"] == ".NET 10.0.12" and not result["flags"]
        assert result["core_sha256"] == receipt["core_sha256"] and result["data_sha256"] == receipt["data_sha256"]
        assert len(result["cases"]) == 21
        rows = []
        for i, row in enumerate(result["cases"]):
            case = audio["cases"][i % 20]
            assert row["name"] == case["name"] and row["repeat"] == (i == 20)
            assert row["matches"] and row["input_and_held_results_unchanged"]
            assert row["pcm_sha256"] == case["pcm_sha256"]
            assert math.isfinite(row["seconds"]) and row["seconds"] > 0
            rows.append(dict(name=row["name"], repeat=row["repeat"], audio_seconds=case["samples"] / 16000,
                             api_seconds=row["seconds"], real_time_factor=row["seconds"] / (case["samples"] / 16000)))
        seconds = math.fsum(row["api_seconds"] for row in rows[:20])
        duration = sum(case["samples"] for case in audio["cases"]) / 16000
        assert duration == 213.265
        records[family] = dict(product_source="c6bf781816f3a45260f7a5eb33fb05d640a821e3",
            result_sha256=sha(base / f"managed-{family}/result.json"),
            process_sha256=sha(base / f"managed-{family}-process.json"),
            audio_seconds=duration, first_pass_requests=20, first_pass_api_seconds=seconds,
            first_pass_real_time_factor=seconds / duration, load_seconds=result["load_seconds"],
            sampled_peak_rss_bytes=job["peak_rss"], rows=rows)

    long_base = root / "artifacts/pyannote-limit-20260919"
    long_receipt = read(long_base / "default/receipt.json")
    long_result = long_base / "default/result.json"
    assert sha(long_result) == long_receipt["files"]["default/result.json"]
    data = read(long_result)
    assert data["passed"] and data["samples"] == 9600000 and data["copies"] == 20
    assert data["full_request_seconds"] == long_receipt["api_request_seconds"]
    pyannote = dict(product_source=long_receipt["product_source"],
        receipt_sha256=sha(long_base / "default/receipt.json"), result_sha256=sha(long_result),
        requests=1, audio_seconds=600, api_seconds=data["full_request_seconds"],
        real_time_factor=data["full_request_seconds"] / 600,
        peak_working_set_bytes=long_receipt["peak_working_set"], windows=long_receipt["windows"],
        affinity=None, affinity_note="This resource probe neither enforces nor records CPU affinity; it is not a single-CPU comparison",
        workload="Twenty repetitions of the same thirty-second dialogue; synthetic resource observation")
    return dict(schema=1, date="2026-09-19", scope="Retrospective descriptive summary of retained API stopwatch measurements; no new inference or matched native latency comparison",
        timing_boundary="Public PCM Transcribe/Diarize call including managed frontend and decoding/postprocessing; excludes model construction, file IO and external validation",
        runtime=".NET 10.0.12", operating_system="Windows", cpu="Intel Core i7-14700KF",
        asr_affinity=[2], asr_receipt_sha256=sha(receipt_path), core_sha256=receipt["core_sha256"], data_sha256=receipt["data_sha256"],
        asr=records, pyannote=pyannote)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for family, row in result["asr"].items():
        print(family, "API seconds", row["first_pass_api_seconds"], "RTF", row["first_pass_real_time_factor"],
              "load", row["load_seconds"], "first/repeat", row["rows"][0]["api_seconds"], row["rows"][-1]["api_seconds"])
    print("pyannote", result["pyannote"])
