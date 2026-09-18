"""Emit minimal single-op broadcast models mirroring the decoded matmul_runtime_ab template (ir 8, opset 14, typed IO, no initializers). Usage: python gen_broadcast.py (writes sibling model.onnx files)."""
import os

def varint(n):
    out = b''
    while True:
        b = n & 0x7f
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out

def tag(f, w):
    return varint((f << 3) | w)

def rec(f, payload):
    return tag(f, 2) + varint(len(payload)) + payload

def vint(f, v):
    return tag(f, 0) + varint(v)

def string(f, s):
    b = s.encode()
    return tag(f, 2) + varint(len(b)) + b

def dim(v):
    return rec(1, vint(1, v))

def value_info(name, dims):
    return string(1, name) + rec(2, rec(1, vint(1, 1) + rec(2, b''.join(dim(d) for d in dims))))

def node(op, inputs, outputs):
    out = b''.join(string(1, i) for i in inputs)
    out += b''.join(string(2, o) for o in outputs)
    out += string(4, op)
    return out

def model(op, inputs, output_name, out_dims, graph_name):
    guts = rec(1, node(op, [n for n, _ in inputs], [output_name]))
    guts += string(2, graph_name)
    for name, dims in inputs:
        guts += rec(11, value_info(name, dims))
    guts += rec(12, value_info(output_name, out_dims))
    opset = tag(1, 2) + varint(0) + vint(2, 14)
    return vint(1, 8) + rec(7, guts) + rec(8, opset)

CASES = {
    'mul_scalar_e5_30': ('Mul', [('x', [12, 30, 30]), ('s', [])], [12, 30, 30]),
    'add_bcast_e5_30': ('Add', [('x', [1, 12, 30, 30]), ('m', [1, 1, 30, 30])], [1, 12, 30, 30]),
    'add_same_e5_30': ('Add', [('x', [12, 30, 30]), ('m', [12, 30, 30])], [12, 30, 30]),
}

root = os.path.dirname(os.path.abspath(__file__))
for kase, (op, inputs, out_dims) in CASES.items():
    data = model(op, inputs, 'y', out_dims, kase)
    path = os.path.join(root, kase, 'model.onnx')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    print(kase, len(data))