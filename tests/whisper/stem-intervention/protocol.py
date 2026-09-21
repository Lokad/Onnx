"""Reuse matrix arithmetic unchanged; replace only the incoming stem state."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIOR_TOOLS = ROOT/'tests/whisper/matmul-precision'
module_spec = importlib.util.spec_from_file_location('qualified_matrix_precision', PRIOR_TOOLS/'protocol.py')
matrix = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(matrix)
np, pin, read, write, rel, raw, source = (getattr(matrix,n) for n in ('np','pin','read','write','rel','raw','source'))
tensor_array, psutil, absent, calculate, scalar_dots, metric = (getattr(matrix,n) for n in ('tensor_array','psutil','absent','calculate','scalar_dots','metric'))
os, sys, THREADS, LIMITS, MODES, SELECTED, TOOLS, MODEL, DATA = (getattr(matrix,n) for n in ('os','sys','THREADS','LIMITS','MODES','SELECTED','TOOLS','MODEL','DATA'))
BASE = ROOT/'artifacts/whisper-stem-intervention-20260921'
PRIOR = matrix.BASE
PROTOCOL = 'whisper-rounded-reference-stem-v1'
CUT = '/Add_2_output_0'


def select_suffix(model, cut):
    matches = [i for i,node in enumerate(model.graph.node) if cut in node.output]
    assert len(matches) == 1, 'Cut must have exactly one producer'
    start = matches[0]+1
    nodes = list(model.graph.node[start:]); assert nodes
    available = {cut}|{i.name for i in model.graph.initializer}
    used_cut = False
    for node in nodes:
        assert all(name in available for name in node.input), ('Unbound suffix dependency', node.name)
        used_cut |= cut in node.input
        assert not any(name in available for name in node.output), 'Duplicate suffix output'
        available.update(node.output)
    assert used_cut, 'Suffix does not consume the cut state'
    return start, nodes


def narrow_stem(value):
    assert value.dtype == np.float64 and value.shape == (1,1500,1280) and np.isfinite(value).all()
    narrowed = value.astype(np.float32)
    assert np.isfinite(narrowed).all()
    return narrowed


def run_suffix(model, cut, state, mode, capture, validate_dots=True):
    import collections
    assert state.dtype == np.float32 and np.isfinite(state).all()
    start, nodes = select_suffix(model, cut)
    initializers = {i.name:i for i in model.graph.initializer}
    uses = collections.Counter(name for node in nodes for name in node.input)
    values = {cut:state}; records = []; checks = []
    before = raw(state)
    for index,node in enumerate(nodes, start):
        inputs = [values[name] if name in values else tensor_array(initializers[name], MODEL.parent) for name in node.input]
        assert all(value.dtype in [np.float32,np.int64] for value in inputs)
        value = calculate(node,inputs,mode)
        assert value.dtype == np.float32 and len(node.output) == 1
        if validate_dots and node.op_type == 'MatMul' and mode == 'wide-matmul':
            checks.append(dict(node=node.name,checks=scalar_dots(inputs[0],inputs[1],value)))
        records.append(dict(index=index,name=node.name,op=node.op_type,dtype=str(value.dtype),shape=list(value.shape)))
        name = node.output[0]; capture(name,value)
        if uses[name]:
            values[name] = value
        for name in node.input:
            uses[name] -= 1
            if uses[name] == 0:
                values.pop(name,None)
        del inputs,value
    assert not values and all(count == 0 for count in uses.values()) and raw(state) == before
    return records, checks
