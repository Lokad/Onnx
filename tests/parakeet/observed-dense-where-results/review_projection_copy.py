"""Check the shared positional slice against both original and optimized graphs."""
import hashlib
import json
from pathlib import Path
import subprocess
import onnx
from onnx import numpy_helper

from review_current_gap import ROOT, OUT, pin, read

REVISION = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'


def main():
    routes = read(OUT/'projection-routes-20260924.json')
    assert routes['passed'] and not routes['route_counter_mismatches']
    original = ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    native_base = ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924/collected/encoder'
    native_path = native_base/'optimized.onnx'
    native_result = read(native_base/'result.json')
    assert native_result['source_model'] == pin(original)
    assert native_result['optimized_model'] == pin(native_path)
    source = onnx.load(original,load_external_data=False)
    native = onnx.load(native_path,load_external_data=False)
    edge = '/pos_enc/Slice_output_0'
    def projections(model):
        nodes = [n for n in model.graph.node if n.name.endswith('/self_attn/linear_pos/MatMul')]
        assert len(nodes) == 24 and all(n.op_type == 'MatMul' and n.input[0] == edge for n in nodes)
        return nodes
    assert {n.name for n in projections(source)} == {n.name for n in projections(native)}
    source_nodes = {n.name:n for n in source.graph.node}
    parent, = [n for n in source.graph.node if 'onnx::Slice_780' in n.output]
    tensor, = [a.t for a in parent.attribute if a.HasField('t')]
    assert parent.op_type == 'Constant' and list(tensor.dims) == [1,9999,1024] and tensor.data_type == 1
    packed_constant, = [t for t in native.graph.initializer if t.name == 'onnx::Slice_780']
    assert list(packed_constant.dims) == list(tensor.dims) and packed_constant.data_type == tensor.data_type
    constants = {}
    for number in ['', '_1','_2','_3','_6']:
        name = '/pos_enc/Constant'+number
        value, = [a.t for a in source_nodes[name].attribute if a.HasField('t')]
        assert value.data_type == 7 and len(value.dims) <= 1
        array = numpy_helper.to_array(value); assert array.size == 1
        constants[name] = int(array.reshape(-1)[0])
    assert list(constants.values()) == [5000,5000,1,1,1]
    assert list(source_nodes['/pos_enc/Sub'].input) == ['/pos_enc/Constant_output_0','/Gather_output_0']
    assert list(source_nodes['/pos_enc/Add'].input) == ['/pos_enc/Constant_1_output_0','/Gather_output_0']
    assert list(source_nodes['/pos_enc/Sub_1'].input) == ['/pos_enc/Add_output_0','/pos_enc/Constant_2_output_0']
    slices = [n for n in native.graph.node if edge in n.output]
    assert len(slices) == 1 and slices[0].op_type == 'Slice'
    managed_sources = ['Tensor.cs','TensorSlice.cs','TensorOps.MatMul.cs','TensorOps.Shape.cs','CPUExecutionProvider.Shape.cs']
    text = {name:(ROOT/'src/Lokad.Onnx'/name).read_text(encoding='utf8') for name in managed_sources}
    assert 'public override DenseTensor<T> ToDenseTensor()' not in text['TensorSlice.cs']
    assert 'return CountedCopy(t.ToDenseTensor(), copy);' in text['TensorOps.MatMul.cs']
    assert 'foreach (var index in denseTensor.GetDimensionsIterator())' in text['Tensor.cs']
    assert 'if (TryCopyContiguousSlice(out var dense)) return dense.Reshape(dimensions);' in text['TensorSlice.cs']
    native_source = 'onnxruntime/core/providers/cpu/tensor/slice.cc'
    raw = subprocess.run(['git','-C',str(ROOT/'external/onnxruntime'),'show',REVISION+':'+native_source],
        check=True,capture_output=True).stdout
    assert b'auto& output_tensor = *ctx->Output(0, output_shape);' in raw
    assert b'output = slice_input_iterator.CopyContiguousInnermostAxes(output);' in raw
    positional = [r for r in routes['aggregates'] if r['key'][:2] == ['positional-route',True]]
    summary = {key:sum(r[key] for r in positional) for key in ['managed_seconds','earlier_ort_seconds',
        'diagnostic_difference','scratch_bytes_per_corpus','copy_bytes_per_corpus']}
    result = dict(passed=True,no_inference=True,shared_edge=edge,consumers=24,
        source_model=pin(original),native_optimized_model=pin(native_path),native_serialization=pin(native_base/'result.json'),
        routes=pin(OUT/'projection-routes-20260924.json'),parent=dict(node=parent.name,shape=list(tensor.dims),dtype='float32'),
        slice=dict(axis=1,step=1,start='5000-T',end_exclusive='5000+T-1',rows='2*T-1'),constants=constants,
        observed_input_materializations_per_corpus=480,summary=summary,
        managed_sources={name:pin(ROOT/'src/Lokad.Onnx'/name) for name in managed_sources},
        ort_source=dict(revision=REVISION,path=native_source,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
        source_conclusion='ORT materializes one shared slice; Lokad MatMul requests independently owned dense input from the shared TensorSlice 24 times. TensorSlice inherits coordinate-by-coordinate ToDenseTensor; its existing contiguous copy helper is called only from Reshape.',
        copy_time_separately_measured=False,whole_application_gain_claimed=False,reviewer=pin(Path(__file__)))
    with (OUT/'projection-copy-20260924.json').open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
