import json
import os
import numpy as np
import pytest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
ARTIFACTS = os.path.join(REPO, "artifacts", "e5")
MODEL = os.path.join(REPO, "models", "multilingual-e5-small", "model.onnx")
TOKENIZER = os.path.join(REPO, "models", "multilingual-e5-small", "sentencepiece.bpe.model")
CASES = os.path.join(REPO, "tests", "e5", "cases.json")

RTOL = 1e-5
ATOL = 1e-6
HIDDEN_RTOL = 2e-4
HIDDEN_ATOL = 1e-5
# Calibrated 2026-09-04: cross-engine float32 noise floor measured at abs<=3.1e-06, rel<=9.7e-05 (all modes);
# the known MatMul kernel defect produced abs err 1152, so this gate keeps 5+ orders of detection headroom.
# Embeddings (the consumed quantity) stay at the strict RTOL/ATOL below.
MIN_SEMANTIC_MARGIN = 0.10

MODES = ["scalar", "simd", "intrinsics"]


processor_holder = {}


def load_tokenizer():
    import sentencepiece as spm
    if 'sp' not in processor_holder:
        processor_holder['sp'] = spm.SentencePieceProcessor(model_file=TOKENIZER)
    return processor_holder['sp']


def load_cases():
    with open(CASES, encoding="utf-8") as f:
        data = json.load(f)
    cases = sorted(data["cases"], key=lambda c: c["id"])
    return cases


def load_dotnet(mode):
    path = os.path.join(ARTIFACTS, "dotnet-%s.json" % mode)
    if not os.path.exists(path):
        pytest.fail("missing .NET artifact for mode %s: %s" % (mode, path))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tensors_by_name(case_entry):
    return {t["name"]: t for t in case_entry["tensors"]}


def tokenize_case(tok, text):
    pieces = tok.encode(text, out_type=int)
    # XLM-R fairseq framing, validated against the Hugging Face reference
    # tokenizer (exact ID match on Hello world and the accented case):
    # bos=0, piece ids shifted by +1, eos=2, keep the first 510 pieces.
    framed = [0] + [p + 1 for p in pieces[:510]] + [2]
    ids = np.array([framed], dtype=np.int64)
    mask = np.ones_like(ids)
    types = np.zeros_like(ids)
    return ids, mask, types
def pooled_embedding(hidden, mask):
    w = mask.astype(np.float64)
    denom = w.sum()
    if denom == 0:
        denom = 1.0
    pooled = (hidden.astype(np.float64) * w[..., None]).sum(axis=1) / denom
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return pooled / norm


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def oracle():
    import onnxruntime as ort
    tok = load_tokenizer()
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    return tok, sess


@pytest.mark.parametrize("mode", MODES)
def test_tokenizer_inputs_exact(oracle, mode):
    tok, _ = oracle
    dotnet = load_dotnet(mode)
    assert dotnet["schemaVersion"] == 1
    for entry in dotnet["cases"]:
        text = next(c["text"] for c in load_cases() if c["id"] == entry["id"])
        ids, mask, types = tokenize_case(tok, text)
        got = tensors_by_name(entry)
        for name, arr in (("input_ids", ids), ("attention_mask", mask), ("token_type_ids", types)):
            t = got[name]
            assert t["dtype"] == "int64", (entry["id"], name)
            assert list(t["shape"]) == list(arr.shape), (entry["id"], name, t["shape"], arr.shape)
            assert list(t["values"]) == arr.flatten().tolist(), (entry["id"], name)


@pytest.mark.parametrize("mode", MODES)
def test_hidden_state_close(oracle, mode):
    tok, sess = oracle
    dotnet = load_dotnet(mode)
    worst_abs = 0.0
    worst_rel = 0.0
    for entry in dotnet["cases"]:
        text = next(c["text"] for c in load_cases() if c["id"] == entry["id"])
        ids, mask, types = tokenize_case(tok, text)
        out = sess.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
        )[0]
        assert np.all(np.isfinite(out)), entry["id"]
        got = tensors_by_name(entry)["last_hidden_state"]
        assert got["dtype"] == "float32"
        assert list(got["shape"]) == list(out.shape), (entry["id"], got["shape"], out.shape)
        ref = np.array(got["values"], dtype=np.float64).reshape(out.shape)
        np.testing.assert_allclose(out, ref, rtol=HIDDEN_RTOL, atol=HIDDEN_ATOL, err_msg=entry["id"])
        worst_abs = max(worst_abs, float(np.max(np.abs(out - ref))))
        denom = np.maximum(np.abs(ref), 1e-12)
        worst_rel = max(worst_rel, float(np.max(np.abs(out - ref) / denom)))
    print("mode=%s max_abs=%.3g max_rel=%.3g" % (mode, worst_abs, worst_rel))


@pytest.mark.parametrize("mode", MODES)
def test_embeddings_close_and_semantic(oracle, mode):
    tok, sess = oracle
    dotnet = load_dotnet(mode)
    embs = {}
    for entry in dotnet["cases"]:
        text = next(c["text"] for c in load_cases() if c["id"] == entry["id"])
        ids, mask, types = tokenize_case(tok, text)
        out = sess.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
        )[0]
        got = tensors_by_name(entry)["embedding"]
        ref = np.array(got["values"], dtype=np.float64)
        mine = pooled_embedding(out, mask)[0]
        np.testing.assert_allclose(mine, ref, rtol=RTOL, atol=ATOL, err_msg=entry["id"])
        embs[entry["id"]] = mine
    related = cosine(embs["query-related"], embs["passage-related"])
    unrelated = cosine(embs["query-related"], embs["passage-unrelated"])
    margin = related - unrelated
    print("mode=%s related=%.4f unrelated=%.4f margin=%.4f" % (mode, related, unrelated, margin))
    assert related > unrelated, (related, unrelated)
    assert margin > MIN_SEMANTIC_MARGIN, margin
