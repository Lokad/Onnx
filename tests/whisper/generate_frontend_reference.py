"""Freeze independent Whisper NumPy/Torch references. Development-only dependencies.

python tests/whisper/generate_frontend_reference.py --output artifacts/whisper-frontend-reference \
    --fixture tests/Lokad.Onnx.Backend.Tests/fixtures/whisper-frontend.json
Requires numpy==2.2.4, transformers==5.16.1, torch==2.11.0 (CPU).
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor


def samples(kind, length):
    values = np.zeros(length, dtype=np.float32)
    if kind == "impulse":
        values[0] = .5
    elif kind == "edges":
        for index, value in [(0, .5), (199, -.25), (200, .75), (479800, -.5), (479999, .25)]:
            values[index] = value
    elif kind in ("noise", "quiet"):
        state = 20260918
        for index in range(length):
            state = (1664525 * state + 1013904223) & 0xffffffff
            values[index] = (((state >> 8) & 65535) - 32768) / 131072
        if kind == "quiet":
            values *= np.float32(1e-6)
    elif kind == "tones":
        time = np.arange(length, dtype=np.float64) / 16000
        values[:] = .2 * np.sin(2 * np.pi * 440 * time) + .1 * np.sin(2 * np.pi * 1000 * time) + .05 * np.sin(2 * np.pi * 3100 * time)
    elif kind == "dc":
        values.fill(.25)
    else:
        raise ValueError(kind)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    assert (np.__version__, transformers.__version__, torch.__version__.split("+")[0]) == ("2.2.4", "5.16.1", "2.11.0")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400, dither=0.0)
    frames = [0, 1, 2, 3, 24, 25, 26, 50, 98, 99, 100, 101, 1499, 2998, 2999]
    rows = []
    files = []
    for index, (kind, length) in enumerate([("impulse", 1), ("edges", 480000), ("noise", 481), ("noise", 480013),
                                           ("quiet", 4099), ("tones", 16000), ("dc", 480000)]):
        audio = samples(kind, length)
        padded = np.zeros((1, 480000), dtype=np.float32)
        padded[0, :min(length, 480000)] = audio[:480000]
        expected = extractor._np_extract_fbank_features(padded, "cpu")
        alternate = extractor._torch_extract_fbank_features(padded, "cpu")
        error = np.abs(expected.astype(np.float64) - alternate)
        name = f"{index:02d}-{kind}-{length}"
        for suffix, array in [("pcm", audio), ("features", expected), ("torch", alternate)]:
            path = args.output / f"{name}-{suffix}.npy"
            np.save(path, array, allow_pickle=False)
            files.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size))
        rows.append(dict(name=name, kind=kind, length=length, frames=frames, values=expected[0, :, frames].tolist(),
                         numpy_vs_torch_max=float(error.max())))
        print(name, "numpy-vs-torch", float(error.max()), flush=True)
    provenance = dict(numpy=np.__version__, transformers=transformers.__version__, torch=torch.__version__,
                      extractor_source_sha256=hashlib.sha256(Path(inspect.getfile(WhisperFeatureExtractor)).read_bytes()).hexdigest(),
                      generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.fixture.write_text(json.dumps(dict(provenance=provenance, absolute_tolerance=1e-5, cases=rows), indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(dict(provenance=provenance, files=files, cases=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
