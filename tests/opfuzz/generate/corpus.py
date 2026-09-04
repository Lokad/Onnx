"""Seeded single-op differential corpus generator (explicit maintenance action).

Reads nothing, writes tests/opfuzz/corpus/<case>/{model.onnx,in_*.txt,ref_*.txt,meta.json}.
Run with the generator env (tests/opfuzz/generate/requirements-generator.txt).
Deterministic for a fixed env: same seed, same files. Review regenerations with git diff.
"""
import json, os, random
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

SEED = 20260904
# Cases where the frozen ORT reference knowingly diverges from the ONNX spec
# (and from Lokad.Onnx, which follows the spec). Recorded into meta.json as
# "known_divergence"; the conformance test xfails these instead of going red.
KNOWN_DIVERGENCES = {
    # Full keepdims=0 reduction over input [2,1]: spec result is scalar ();
    # ORT 1.29 returns (1,) whenever a size-1 dim is present (verified [] on
    # full reductions without size-1 dims). Values agree bit-identically.
    "reducemean_2": "ORT returns (1,) for full keepdims=0 reduction over a size-1 dim; spec and Lokad.Onnx give scalar ()",
}
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corpus")
rng = random.Random(SEED)
npr = np.random.default_rng(SEED)
OPSET = 14
ENV = {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__}

def txt_save(path, arr):
    arr = np.ascontiguousarray(arr)
    dtype = {np.dtype("float32"): "float32", np.dtype("float64"): "float64", np.dtype("int64"): "int64"}[arr.dtype]
    with open(path, "w") as f:
        f.write("%s %d %s\n" % (dtype, arr.ndim, " ".join(str(d) for d in arr.shape)))
        f.write(" ".join(repr(float(v)) if arr.dtype != np.dtype("int64") else str(int(v)) for v in arr.ravel()))
        f.write("\n")

def eshape(rank, max_dim=6):
    return [rng.randint(1, max_dim) for _ in range(rank)]

def rshape(max_rank=3, max_dim=6):
    return eshape(rng.randint(1, max_rank), max_dim)

def rarray(shape, lo=-3.0, hi=3.0):
    return (npr.uniform(lo, hi, size=shape)).astype(np.float32)

def bshape(a, b):
    return list(np.broadcast_shapes(tuple(a), tuple(b)))

def run_ort(mp, feed):
    sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
    return sess.run(None, {n: np.ascontiguousarray(a, dtype=np.float32) for n, a in feed.items()})

def write_model(d, case_id, node, inputs, out_shapes, inits):
    vin = [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(s)) for n, s in inputs]
    vout = [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(s)) for n, s in out_shapes]
    g = helper.make_graph([node], "g_" + case_id, vin, vout, initializer=list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="opfuzz")
    m.ir_version = 8
    mp = os.path.join(d, "model.onnx")
    onnx.save(m, mp)
    return mp

def emit(case_id, node, inputs, out_shapes, feed, inits=()):
    d = os.path.join(ROOT, case_id)
    os.makedirs(d, exist_ok=True)
    mp = write_model(d, case_id, node, inputs, out_shapes, inits)
    res = run_ort(mp, feed)
    drifted = [(n, list(a.shape)) for (n, s), a in zip(out_shapes, res) if list(a.shape) != list(s)]
    if drifted:
        out_shapes = [(n, list(a.shape)) for (n, s), a in zip(out_shapes, res)]
        mp = write_model(d, case_id, node, inputs, out_shapes, inits)
        res = run_ort(mp, feed)
        still = [(n, list(a.shape)) for (n, s), a in zip(out_shapes, res) if list(a.shape) != list(s)]
        if still:
            print("SKIP %s: reference shape unstable %s" % (case_id, still))
            return None
        print("note %s: declared shape corrected to %s" % (case_id, out_shapes))
    for n, a in feed.items():
        txt_save(os.path.join(d, "in_" + n + ".txt"), a)
    for (n, s), a in zip(out_shapes, res):
        assert np.all(np.isfinite(a)), "non-finite ORT output in " + case_id
        txt_save(os.path.join(d, "ref_" + n + ".txt"), a)
    with open(os.path.join(d, "meta.json"), "w") as f:
        meta = {"case": case_id, "seed": SEED, "opset": OPSET, "ir": 8, "env": ENV}
        if case_id in KNOWN_DIVERGENCES:
            meta["known_divergence"] = KNOWN_DIVERGENCES[case_id]
        json.dump(meta, f, indent=1)
    return case_id

def binary_cases(op, n=8):
    for i in range(n):
        s = rshape()
        a = rarray(s)
        r = rng.random()
        if r < 0.4:
            b = rarray(s)
        elif r < 0.7:
            b = rarray([s[-1]])
        else:
            b = rarray([1])
        if op == "Div":
            b = b + np.copysign(2.0, b).astype(np.float32)
        node = helper.make_node(op, ["x", "y"], ["z"])
        emit("%s_%d" % (op.lower(), i), node, [("x", list(a.shape)), ("y", list(b.shape))],
             [("z", bshape(a.shape, b.shape))], {"x": a, "y": b})

def unary_cases(op, n=5, lo=-3.0, hi=3.0):
    for i in range(n):
        s = rshape()
        a = rarray(s, lo, hi)
        node = helper.make_node(op, ["x"], ["z"])
        emit("%s_%d" % (op.lower(), i), node, [("x", list(a.shape))], [("z", list(a.shape))], {"x": a})

def transpose_cases(n=5):
    for i in range(n):
        rank = rng.randint(2, 4)
        s = eshape(rank, 5)
        a = rarray(s)
        if i % 2 == 0:
            perm = list(range(rank)); rng.shuffle(perm)
            node = helper.make_node("Transpose", ["x"], ["z"], perm=perm)
            zs = [s[p] for p in perm]
        else:
            node = helper.make_node("Transpose", ["x"], ["z"])
            zs = list(reversed(s))
        emit("transpose_%d" % i, node, [("x", list(a.shape))], [("z", zs)], {"x": a})

def reshape_cases(n=5):
    for i in range(n):
        s = rshape(3, 4)
        a = rarray(s)
        vol = int(np.prod(s))
        fac = [d for d in range(2, 7) if vol % d == 0]
        if fac and i % 2 == 0:
            d = rng.choice(fac)
            ns = [d, vol // d]
        else:
            ns = [vol // 2, 2] if vol % 2 == 0 else [vol]
        shape_init = helper.make_tensor("shape", TensorProto.INT64, [len(ns)], np.array(ns, dtype=np.int64))
        node = helper.make_node("Reshape", ["x", "shape"], ["z"])
        emit("reshape_%d" % i, node, [("x", list(a.shape))], [("z", ns)], {"x": a}, inits=[shape_init])

def concat_cases(n=5):
    for i in range(n):
        rank = rng.randint(1, 3)
        base = eshape(rank, 4)
        ax = rng.randrange(-rank, rank)
        d0 = rng.randint(1, 4); d1 = rng.randint(1, 4)
        s0 = list(base); s1 = list(base); s0[ax] = d0; s1[ax] = d1
        zs = list(s0); zs[ax] = d0 + d1
        node = helper.make_node("Concat", ["x", "y"], ["z"], axis=ax)
        emit("concat_%d" % i, node, [("x", s0), ("y", s1)], [("z", zs)], {"x": rarray(s0), "y": rarray(s1)})

def softmax_cases(n=5):
    for i in range(n):
        rank = rng.randint(1, 4)
        s = eshape(rank, 5)
        ax = rng.choice([-1, -1, 0, rank - 1])
        node = helper.make_node("Softmax", ["x"], ["z"], axis=ax)
        emit("softmax_%d" % i, node, [("x", list(s))], [("z", list(s))], {"x": rarray(s, -5, 5)})

def matmul_cases():
    cfgs = [(1, 1, 1), (1, 5, 1), (2, 3, 4), (3, 3, 32), (4, 3, 32), (5, 31, 7),
            (2, 32, 2), (4, 33, 5), (3, 8, 8), (6, 16, 16), (1, 64, 1)]
    for i, (m, k, n_) in enumerate(cfgs):
        a = rarray([m, k], -1, 1); b = rarray([k, n_], -1, 1)
        node = helper.make_node("MatMul", ["x", "y"], ["z"])
        emit("matmul_%d" % i, node, [("x", [m, k]), ("y", [k, n_])], [("z", [m, n_])], {"x": a, "y": b})
    a = rarray([2, 3, 4], -1, 1); b = rarray([2, 4, 5], -1, 1)
    node = helper.make_node("MatMul", ["x", "y"], ["z"])
    emit("matmul_batched", node, [("x", [2, 3, 4]), ("y", [2, 4, 5])], [("z", [2, 3, 5])], {"x": a, "y": b})

def reducemean_cases(n=5):
    for i in range(n):
        rank = rng.randint(2, 4)
        s = eshape(rank, 5)
        a = rarray(s)
        nax = rng.randint(1, rank)
        axes = rng.sample(range(rank), nax)
        if rng.random() < 0.3:
            axes = [a_ - rank for a_ in axes]
        kd = rng.randint(0, 1)
        norm = sorted(a_ if a_ >= 0 else a_ + rank for a_ in axes)
        zs = [1 if (j in norm and kd) else d for j, d in enumerate(s)]
        if not kd:
            zs = [d for j, d in enumerate(s) if j not in norm] or []
        node = helper.make_node("ReduceMean", ["x"], ["z"], axes=axes, keepdims=kd)
        emit("reducemean_%d" % i, node, [("x", list(s))], [("z", zs)], {"x": a})

def unsqueeze_cases(n=5):
    for i in range(n):
        rank = rng.randint(1, 3)
        s = eshape(rank, 4)
        a = rarray(s)
        nax = rng.randint(1, 2)
        axes = rng.sample(range(-rank - nax, rank + nax), nax)
        ax_init = helper.make_tensor("axes", TensorProto.INT64, [len(axes)], np.array(axes, dtype=np.int64))
        node = helper.make_node("Unsqueeze", ["x", "axes"], ["z"])
        try:
            zs = list(s)
            norm = sorted(a_ if a_ >= 0 else a_ + len(zs) + 1 for a_ in axes)
            for a_ in norm:
                zs.insert(a_, 1)
            if emit("unsqueeze_%d" % i, node, [("x", list(s))], [("z", zs)], {"x": a}, inits=[ax_init]) is None:
                print("skip unsqueeze_%d (unstable)" % i)
        except Exception as e:
            print("skip unsqueeze_%d (%s)" % (i, e))

def squeeze_cases(n=4):
    for i in range(n):
        s = [rng.choice([1, 3]), rng.choice([1, 4]), rng.choice([1, 2])]
        a = rarray(s)
        ones = [j for j, d in enumerate(s) if d == 1]
        axes = rng.sample(ones, min(len(ones), rng.randint(1, 2))) if ones else []
        zs = [d for j, d in enumerate(s) if j not in axes]
        if axes:
            ax_init = helper.make_tensor("axes", TensorProto.INT64, [len(axes)], np.array(axes, dtype=np.int64))
            node = helper.make_node("Squeeze", ["x", "axes"], ["z"])
            emit("squeeze_%d" % i, node, [("x", list(s))], [("z", zs)], {"x": a}, inits=[ax_init])
        else:
            node = helper.make_node("Squeeze", ["x"], ["z"])
            emit("squeeze_%d" % i, node, [("x", list(s))], [("z", zs)], {"x": a})

def gather_cases(n=4):
    for i in range(n):
        rank = rng.randint(1, 3)
        s = eshape(rank, 5)
        a = rarray(s)
        ax = rng.randrange(0, rank)
        ni = rng.randint(1, 4)
        idx = npr.integers(-s[ax], s[ax], size=[ni]).astype(np.int64)
        idx_init = helper.make_tensor("idx", TensorProto.INT64, [ni], idx)
        node = helper.make_node("Gather", ["x", "idx"], ["z"], axis=ax)
        zs = list(s[:ax]) + [ni] + list(s[ax + 1:])
        d = os.path.join(ROOT, "gather_%d" % i)
        os.makedirs(d, exist_ok=True)
        vin = [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(s))]
        vout = [helper.make_tensor_value_info("z", TensorProto.FLOAT, zs)]
        g = helper.make_graph([node], "g", vin, vout, initializer=[idx_init])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="opfuzz")
        m.ir_version = 8
        mp = os.path.join(d, "model.onnx")
        onnx.save(m, mp)
        txt_save(os.path.join(d, "in_x.txt"), a)
        res = run_ort(mp, {"x": a})
        assert list(res[0].shape) == zs and np.all(np.isfinite(res[0]))
        txt_save(os.path.join(d, "ref_z.txt"), res[0])
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"case": "gather_%d" % i, "seed": SEED, "opset": OPSET, "ir": 8, "env": ENV}, f, indent=1)

if __name__ == "__main__":
    for op in ["Add", "Sub", "Mul", "Div"]:
        binary_cases(op)
    for op in ["Relu", "Sqrt", "Neg", "Abs"]:
        unary_cases(op, lo=(0.01 if op == "Sqrt" else -3.0))
    transpose_cases(); reshape_cases(); concat_cases(); softmax_cases()
    matmul_cases(); reducemean_cases(); unsqueeze_cases(); squeeze_cases(); gather_cases()
    print("cases:", len([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]))
