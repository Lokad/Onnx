"""Inspect the retained decoder protobuf without loading or executing a runtime."""
import hashlib
import json
import math
from pathlib import Path
import onnx

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
MODEL = ROOT / 'models/parakeet-tdt-0.6b-v3/decoder_joint-model.onnx'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    destination = OUTPUT / 'decoder-lstm-model-20260924.json'
    assert not destination.exists()
    identity = pin(MODEL)
    assert identity == dict(bytes=72520893, sha256='e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1')
    model = onnx.load(MODEL, load_external_data=False)
    initializers = {t.name: t for t in model.graph.initializer}
    graph_io = {v.name for v in [*model.graph.input, *model.graph.output]}
    rows = []; weights = {}
    for node in model.graph.node:
        if node.op_type != 'LSTM': continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert attrs == dict(hidden_size=640)
        names = {}
        for role, index in [('W', 1), ('R', 2)]:
            name = node.input[index]; tensor = initializers[name]
            assert tensor.data_type == onnx.TensorProto.FLOAT and list(tensor.dims) == [1, 2560, 640]
            assert not tensor.external_data and name not in graph_io
            size = math.prod(tensor.dims) * 4; assert len(tensor.raw_data) == size
            consumers = [dict(node=n.name, op=n.op_type, input_index=i)
                         for n in model.graph.node for i, value in enumerate(n.input) if value == name]
            assert consumers == [dict(node=node.name, op='LSTM', input_index=index)]
            names[role] = name
            weights[name] = dict(shape=list(tensor.dims), dtype='Float', bytes=size,
                source_sha256=hashlib.sha256(tensor.raw_data).hexdigest(), consumers=consumers)
        rows.append(dict(name=node.name, inputs=list(node.input), outputs=list(node.output), attributes=attrs, weights=names))
    assert len(rows) == 2 and len(weights) == 4
    raw_bytes = sum(v['bytes'] for v in weights.values()); assert raw_bytes == 26214400
    residency = ROOT / 'tests/parakeet/inclusive-packing-results/residency-observations-20260924.json'
    resident = json.loads(residency.read_text())
    assert resident['passed'] and resident['comparison']['graphs']['decoder']['selected'] == 3
    # This is a capacity calculation from the qualified mapping census, not
    # actual new preparation or a promise that the proposed layout is admitted.
    selected_bytes = 25246720; cap = 67108864
    value = dict(passed=True, scope='Original ONNX metadata and raw constant bytes only; no inference or new prepared map',
        model=identity, onnx_parser_version=onnx.__version__, nodes=rows, weights=weights,
        raw_float_panel_bytes=raw_bytes, qualified_existing_decoder_packing_bytes=selected_bytes,
        hypothetical_total_bytes=selected_bytes + raw_bytes, unchanged_decoder_cap=cap,
        hypothetical_remaining_bytes=cap - selected_bytes - raw_bytes,
        actual_new_residency_qualified=False, source_residency_report=pin(residency), generator=pin(Path(__file__)))
    destination.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(dict(passed=True, nodes=len(rows), weights=len(weights), bytes=raw_bytes, output=pin(destination))))


if __name__ == '__main__': main()
