"""Seeded single-op differential corpus generator (explicit maintenance action).

Reads nothing, writes tests/opfuzz/corpus/<case>/{model.onnx,in_*.txt,ref_*.txt,meta.json}.
Corpus models are tracked in git through a scoped .gitignore exception; keep
them committed alongside any new case instead of relying on regeneration.
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
    arr = np.asarray(arr)
    dtype = {np.dtype("float32"): "float32", np.dtype("float64"): "float64", np.dtype("int64"): "int64", np.dtype("int32"): "int32", np.dtype("bool"): "bool", np.dtype("uint32"): "uint32", np.dtype("uint64"): "uint64", np.dtype("int8"): "int8", np.dtype("uint8"): "uint8", np.dtype("int16"): "int16", np.dtype("uint16"): "uint16"}[arr.dtype]
    with open(path, "w") as f:
        f.write("%s %d %s\n" % (dtype, arr.ndim, " ".join(str(d) for d in arr.shape)))
        if arr.dtype == np.dtype("bool"):
            f.write(" ".join("1" if v else "0" for v in arr.ravel()))
        else:
            f.write(" ".join(repr(float(v)) if arr.dtype not in (np.dtype("int64"), np.dtype("int32"), np.dtype("uint32"), np.dtype("uint64"), np.dtype("int8"), np.dtype("uint8"), np.dtype("int16"), np.dtype("uint16")) else str(int(v)) for v in arr.ravel()))
        f.write("\n")

def eshape(rank, max_dim=6):
    return [rng.randint(1, max_dim) for _ in range(rank)]

def rshape(max_rank=3, max_dim=6):
    return eshape(rng.randint(1, max_rank), max_dim)

def rarray(shape, lo=-3.0, hi=3.0):
    return (npr.uniform(lo, hi, size=shape)).astype(np.float32)

def bshape(a, b):
    return list(np.broadcast_shapes(tuple(a), tuple(b)))

def run_ort(mp, feed, dtypes=None):
    dtypes = dtypes or {}
    sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
    return sess.run(None, {n: np.asarray(a, dtype=dtypes.get(n, np.float32)) for n, a in feed.items()})

def write_model(d, case_id, node, inputs, out_shapes, inits, dtypes=None, opset=None):
    dtypes = dtypes or {}
    opset = OPSET if opset is None else opset
    vin = [helper.make_tensor_value_info(n, dtypes.get(n, TensorProto.FLOAT), list(s)) for n, s in inputs]
    vout = [helper.make_tensor_value_info(n, dtypes.get(n, TensorProto.FLOAT), list(s)) for n, s in out_shapes]
    g = helper.make_graph([node], "g_" + case_id, vin, vout, initializer=list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], producer_name="opfuzz")
    m.ir_version = 8
    mp = os.path.join(d, "model.onnx")
    onnx.save(m, mp)
    return mp

def emit(case_id, node, inputs, out_shapes, feed, inits=(), dtypes=None, opset=None, feed_dtypes=None):
    d = os.path.join(ROOT, case_id)
    os.makedirs(d, exist_ok=True)
    mp = write_model(d, case_id, node, inputs, out_shapes, inits, dtypes, opset)
    res = run_ort(mp, feed, feed_dtypes)
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
        meta = {"case": case_id, "seed": SEED, "opset": OPSET if opset is None else opset, "ir": 8, "env": ENV}
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

def softmax_large_cases():
    # Fixed values only (no RNG draws): large-magnitude rows exercise the
    # max-subtraction stability path differentially.
    a = np.array([[1000.0, 1001.0]], dtype=np.float32)
    node = helper.make_node("Softmax", ["x"], ["z"], axis=-1)
    emit("softmax_large", node, [("x", [1, 2])], [("z", [1, 2])], {"x": a})

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
    # Fixed values only (no RNG draws): 64+ row counts route the
    # panel-packed kernel (and the odd-row tail at 65), which no sized
    # random case reaches deterministically.
    pa = (np.arange(64 * 8, dtype=np.float32).reshape(64, 8) % 7) - 3
    pb = (np.arange(8 * 8, dtype=np.float32).reshape(8, 8) % 5) - 2
    node = helper.make_node("MatMul", ["x", "y"], ["z"])
    emit("matmul_packed", node, [("x", [64, 8]), ("y", [8, 8])], [("z", [64, 8])], {"x": pa, "y": pb})
    qa = (np.arange(65 * 8, dtype=np.float32).reshape(65, 8) % 7) - 3
    node = helper.make_node("MatMul", ["x", "y"], ["z"])
    emit("matmul_packed_tail", node, [("x", [65, 8]), ("y", [8, 8])], [("z", [65, 8])], {"x": qa, "y": pb})

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

def layernorm_cases():
    LNOPSET = 20
    F32 = TensorProto.FLOAT
    F64 = TensorProto.DOUBLE

    def finit(name, dtype, shape, vals):
        t = F32 if dtype == np.float32 else F64
        return helper.make_tensor(name, t, list(shape), np.asarray(vals, dtype=dtype).reshape(-1))

    # Default form: no attributes at all, single output.
    a = rarray([2, 4])
    node = helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"])
    emit("layernorm_default", node, [("x", [2, 4])], [("y", [2, 4])], {"x": a},
         inits=[finit("s", np.float32, [4], rarray([4]) + 1.0), finit("b", np.float32, [4], rarray([4]))],
         opset=LNOPSET)
    # Explicit standard form: axis, epsilon and stash_type=1.
    a = rarray([2, 4])
    node = helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"], axis=-1, epsilon=1e-5, stash_type=1)
    emit("layernorm_explicit", node, [("x", [2, 4])], [("y", [2, 4])], {"x": a},
         inits=[finit("s", np.float32, [4], rarray([4]) + 1.0), finit("b", np.float32, [4], rarray([4]))],
         opset=LNOPSET)
    # Non-last axis with all three outputs (Y, Mean, InvStdDev).
    a = rarray([2, 3, 4])
    node = helper.make_node("LayerNormalization", ["x", "s", "b"], ["y", "m", "v"], axis=1, epsilon=1e-3, stash_type=1)
    emit("layernorm_axis1", node, [("x", [2, 3, 4])],
         [("y", [2, 3, 4]), ("m", [2, 1, 1]), ("v", [2, 1, 1])], {"x": a},
         inits=[finit("s", np.float32, [3, 4], rarray([3, 4]) + 1.0), finit("b", np.float32, [3, 4], rarray([3, 4]))],
         opset=LNOPSET)
    # No-bias form with Mean output.
    a = rarray([2, 4])
    node = helper.make_node("LayerNormalization", ["x", "s"], ["y", "m"], axis=-1, epsilon=1e-5, stash_type=1)
    emit("layernorm_nobias", node, [("x", [2, 4])], [("y", [2, 4]), ("m", [2, 1])], {"x": a},
         inits=[finit("s", np.float32, [4], rarray([4]) + 1.0)],
         opset=LNOPSET)
    # Double precision, single output.
    a = npr.uniform(-3.0, 3.0, size=[2, 4]).astype(np.float64)
    node = helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"], axis=-1, epsilon=1e-5, stash_type=1)
    emit("layernorm_double", node, [("x", [2, 4])], [("y", [2, 4])], {"x": a},
         inits=[finit("s", np.float64, [4], npr.uniform(-3.0, 3.0, size=[4]).astype(np.float64) + 1.0),
                finit("b", np.float64, [4], npr.uniform(-3.0, 3.0, size=[4]).astype(np.float64))],
         dtypes={"x": F64, "y": F64}, feed_dtypes={"x": np.float64}, opset=LNOPSET)
    # Double precision with float32 stats outputs.
    a = npr.uniform(-3.0, 3.0, size=[1, 6]).astype(np.float64)
    node = helper.make_node("LayerNormalization", ["x", "s", "b"], ["y", "m", "v"], axis=-1, epsilon=1e-5, stash_type=1)
    emit("layernorm_double_stats", node, [("x", [1, 6])],
         [("y", [1, 6]), ("m", [1, 1]), ("v", [1, 1])], {"x": a},
         inits=[finit("s", np.float64, [6], npr.uniform(-3.0, 3.0, size=[6]).astype(np.float64) + 1.0),
                finit("b", np.float64, [6], npr.uniform(-3.0, 3.0, size=[6]).astype(np.float64))],
         dtypes={"x": F64, "y": F64}, feed_dtypes={"x": np.float64}, opset=LNOPSET)

def boundary_cases():
    # Scalar (rank-0) broadcast against a matrix.
    a = rarray([2, 3])
    b = np.float32(1.5)
    node = helper.make_node("Add", ["x", "y"], ["z"])
    emit("add_scalar", node, [("x", [2, 3]), ("y", [])], [("z", [2, 3])], {"x": a, "y": b})
    # Size-1 axis in the middle, broadcast on both sides.
    a = rarray([2, 1, 4]); b = rarray([2, 3, 4])
    node = helper.make_node("Mul", ["x", "y"], ["z"])
    emit("mul_mid1", node, [("x", [2, 1, 4]), ("y", [2, 3, 4])], [("z", [2, 3, 4])], {"x": a, "y": b})
    a = rarray([3, 1]); b = rarray([1, 4])
    node = helper.make_node("Sub", ["x", "y"], ["z"])
    emit("sub_rowcol", node, [("x", [3, 1]), ("y", [1, 4])], [("z", [3, 4])], {"x": a, "y": b})
    # Reshape with -1 dimension inference.
    a = rarray([2, 6])
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], np.array([-1, 3], dtype=np.int64))
    node = helper.make_node("Reshape", ["x", "shape"], ["z"])
    emit("reshape_neg1", node, [("x", [2, 6])], [("z", [4, 3])], {"x": a}, inits=[shape_init])
    # Three-input concat.
    c0 = rarray([2, 2]); c1 = rarray([2, 2]); c2 = rarray([2, 2])
    node = helper.make_node("Concat", ["x", "y", "w"], ["z"], axis=1)
    emit("concat_3way", node, [("x", [2, 2]), ("y", [2, 2]), ("w", [2, 2])], [("z", [2, 6])],
         {"x": c0, "y": c1, "w": c2})
    # Softmax over large logits (stability boundary).
    a = rarray([2, 4], -50.0, 50.0)
    node = helper.make_node("Softmax", ["x"], ["z"], axis=-1)
    emit("softmax_big", node, [("x", [2, 4])], [("z", [2, 4])], {"x": a})
    # Full reduction with no axes attribute, keepdims=1.
    a = rarray([2, 3])
    node = helper.make_node("ReduceMean", ["x"], ["z"], keepdims=1)
    emit("reducemean_full", node, [("x", [2, 3])], [("z", [1, 1])], {"x": a})
    # Gather on a negative axis with negative indices.
    a = rarray([2, 3, 4])
    idx = npr.integers(-4, 4, size=[3]).astype(np.int64)
    idx_init = helper.make_tensor("idx", TensorProto.INT64, [3], idx)
    node = helper.make_node("Gather", ["x", "idx"], ["z"], axis=-1)
    d = os.path.join(ROOT, "gather_negaxis")
    os.makedirs(d, exist_ok=True)
    vin = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])]
    vout = [helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3, 3])]
    g = helper.make_graph([node], "g", vin, vout, initializer=[idx_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="opfuzz")
    m.ir_version = 8
    mp = os.path.join(d, "model.onnx")
    onnx.save(m, mp)
    txt_save(os.path.join(d, "in_x.txt"), a)
    res = run_ort(mp, {"x": a})
    assert list(res[0].shape) == [2, 3, 3] and np.all(np.isfinite(res[0]))
    txt_save(os.path.join(d, "ref_z.txt"), res[0])
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"case": "gather_negaxis", "seed": SEED, "opset": OPSET, "ir": 8, "env": ENV}, f, indent=1)

def gather_extra_cases():
    # Higher-rank int64 indices over 1-D data (C02 reproducer).
    a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    idx = np.array([[0, 2], [1, 0]], dtype=np.int64)
    idx_init = helper.make_tensor("idx", TensorProto.INT64, [2, 2], idx)
    node = helper.make_node("Gather", ["x", "idx"], ["z"], axis=0)
    emit("gather_highrank", node, [("x", [3])], [("z", [2, 2])], {"x": a}, inits=[idx_init])
    # Scalar int64 index.
    b = np.arange(6, dtype=np.float32).reshape(2, 3)
    sidx_init = helper.make_tensor("idx", TensorProto.INT64, [], np.array(1, dtype=np.int64))
    node = helper.make_node("Gather", ["x", "idx"], ["z"], axis=0)
    emit("gather_scalar", node, [("x", [2, 3])], [("z", [3])], {"x": b}, inits=[sidx_init])
    # Higher-rank int32 indices.
    c = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    i32 = np.array([[2, 0], [1, 1]], dtype=np.int32)
    i32_init = helper.make_tensor("idx", TensorProto.INT32, [2, 2], i32)
    node = helper.make_node("Gather", ["x", "idx"], ["z"], axis=1)
    emit("gather_int32", node, [("x", [2, 3, 4])], [("z", [2, 2, 2, 4])], {"x": c}, inits=[i32_init])

def conv_basic_cases():
    # Fixed values only: no shared RNG draws, so existing cases are unaffected.
    x = np.arange(50, dtype=np.float32).reshape(1, 2, 5, 5)
    w = (np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3) % 5) - 2.0
    w_init = helper.make_tensor("w", TensorProto.FLOAT, [2, 2, 3, 3], w.astype(np.float32))
    node = helper.make_node("Conv", ["x", "w"], ["z"], kernel_shape=[3, 3])
    emit("conv_basic_3x3", node, [("x", [1, 2, 5, 5])], [("z", [1, 2, 3, 3])], {"x": x}, inits=[w_init])
    x1 = np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4)
    w1 = (np.arange(6, dtype=np.float32).reshape(2, 3, 1, 1) % 5) - 2.0
    w1_init = helper.make_tensor("w", TensorProto.FLOAT, [2, 3, 1, 1], w1.astype(np.float32))
    node = helper.make_node("Conv", ["x", "w"], ["z"], kernel_shape=[1, 1])
    emit("conv_1x1", node, [("x", [1, 3, 4, 4])], [("z", [1, 2, 4, 4])], {"x": x1}, inits=[w1_init])


def shape_extra_cases():
    # Reversed slice yields an empty shape vector (C03).
    a = np.zeros((2, 3, 4), dtype=np.float32)
    node = helper.make_node("Shape", ["x"], ["z"], start=2, end=1)
    emit("shape_empty", node, [("x", [2, 3, 4])], [("z", [0])], {"x": a}, dtypes={"z": TensorProto.INT64}, opset=15)
    # Negative sliced bounds.
    node = helper.make_node("Shape", ["x"], ["z"], start=-2, end=-1)
    emit("shape_negative", node, [("x", [2, 3, 4])], [("z", [1])], {"x": a}, dtypes={"z": TensorProto.INT64}, opset=15)
    # No slice: full shape on the current opset.
    node = helper.make_node("Shape", ["x"], ["z"])
    emit("shape_full", node, [("x", [2, 3, 4])], [("z", [3])], {"x": a}, dtypes={"z": TensorProto.INT64}, opset=15)

if __name__ == "__main__":
    for op in ["Add", "Sub", "Mul", "Div"]:
        binary_cases(op)
    for op in ["Relu", "Sqrt", "Neg", "Abs"]:
        unary_cases(op, lo=(0.01 if op == "Sqrt" else -3.0))
    transpose_cases(); reshape_cases(); concat_cases(); softmax_cases(); softmax_large_cases()
    matmul_cases(); reducemean_cases(); unsqueeze_cases(); squeeze_cases(); gather_cases()
    boundary_cases()
    layernorm_cases()
    gather_extra_cases()
    shape_extra_cases()
    conv_basic_cases()
    print("cases:", len([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]))
