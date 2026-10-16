"""Independent, file-based inputs and numerical contracts for the Gemma lane."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formatted_text(case: dict) -> str:
    text = case["text"] * case.get("repeat", 1)
    role = case["role"]
    if role == "query":
        return "task: search result | query: " + text
    if role == "document":
        return f"title: {case.get('title') or 'none'} | text: {text}"
    if role == "raw":
        return text
    raise ValueError(f"Unknown role {role!r}")


def save_array(directory: Path, name: str, array: np.ndarray) -> dict:
    path = directory / (name + ".npy")
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)
    return {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": sha256(path)}


def pooled(hidden: np.ndarray, mask: np.ndarray, dimensions: int = 768) -> np.ndarray:
    if dimensions not in (128, 256, 512, 768):
        raise ValueError("Dimensions must be 128, 256, 512, or 768")
    valid = np.asarray(mask != 0, dtype=np.float32)[..., None]
    mean = np.sum(np.where(valid != 0, hidden, np.float32(0)), axis=1, dtype=np.float32)
    mean /= np.maximum(valid.sum(axis=1, dtype=np.float32), np.float32(1e-9))
    shortened = mean[:, :dimensions]
    norm = np.sqrt(np.sum(shortened * shortened, axis=-1, keepdims=True, dtype=np.float32))
    if not np.isfinite(shortened).all() or not np.isfinite(norm).all() or np.any(norm <= 1e-12):
        raise ValueError("Nonfinite or zero-norm embedding")
    return shortened / norm


def numerical_check(actual: np.ndarray, reference: np.ndarray, *, components: bool) -> dict:
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise AssertionError(f"Tensor contract differs: {actual.shape}/{actual.dtype} vs {reference.shape}/{reference.dtype}")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise AssertionError("Nonfinite tensor")
    a = actual.astype(np.float64)
    r = reference.astype(np.float64)
    delta = np.abs(a - r)
    relative_l2 = float(np.linalg.norm(a - r) / max(float(np.linalg.norm(r)), 1e-12))
    result = {"maxAbsoluteError": float(delta.max(initial=0)), "relativeL2": relative_l2}
    if relative_l2 > 1e-4:
        raise AssertionError(f"Relative L2 exceeds 1e-4: {result}")
    if components:
        allowance = 1e-6 + 1e-5 * np.abs(r)
        ratio = float((delta / allowance).max(initial=0))
        result["maxComponentAllowanceRatio"] = ratio
        cosine = np.sum(a * r, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(r, axis=-1))
        result["minimumCosine"] = float(cosine.min())
        if ratio > 1 or np.any(cosine < 0.99999):
            raise AssertionError(f"Embedding numerical gate failed: {result}")
    return result


def verify_assets(model_dir: Path, kind: str = "onnx") -> dict:
    manifest_path = Path(__file__).resolve().parents[1] / "assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))[kind]
    for name, expected in manifest["files"].items():
        path = model_dir / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or mismatched pinned asset: {path}")
    return manifest


def session(path: Path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
