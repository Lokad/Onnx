"""Generate full Parakeet component references with independently carried native states."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument, RuntimeException


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def tensors(graph):
    yield from graph.initializer
    for item in graph.sparse_initializer:
        yield item.values
        yield item.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from tensors(nested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert (np.__version__, onnx.__version__, ort.__version__) == ('2.2.4', '1.22.0', '1.29.0')
    args.output.mkdir(parents=True, exist_ok=False)
    assets_path = Path(__file__).with_name('assets.json')
    assets = json.loads(assets_path.read_text(encoding='utf-8'))
    for name, item in assets['files'].items():
        path = args.models / name
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
    # Validate every external tensor's local path and range, not only a filename convention.
    references = []
    for name in assets['graphs'].values():
        graph = onnx.load(args.models / name, load_external_data=False)
        for tensor in tensors(graph.graph):
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            data = {entry.key: entry.value for entry in tensor.external_data}
            path = (args.models / data['location']).resolve()
            assert path.is_relative_to(args.models.resolve()) and data['location'] in assets['files']
            offset = int(data.get('offset', '0'))
            length = int(data.get('length', str(path.stat().st_size - offset)))
            assert 0 <= offset <= offset + length <= path.stat().st_size
            references.append(dict(model=name, tensor=tensor.name, **data))
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    sessions = {name: ort.InferenceSession(str(args.models / path), options, providers=['CPUExecutionProvider'])
                for name, path in assets['graphs'].items()}
    files, scenarios = {}, []

    def save(name, values):
        values = np.ascontiguousarray(values)
        assert values.dtype in (np.dtype('float32'), np.dtype('int32'), np.dtype('int64'))
        assert np.isfinite(values).all()
        path = args.output / (name + '.npy')
        assert not path.exists()
        np.save(path, values, allow_pickle=False)
        files[path.name] = dict(dtype=str(values.dtype), shape=list(values.shape), bytes=path.stat().st_size, sha256=sha(path))
        return path.name

    def run(scenario, model, inputs, bindings, failure=False, repeat=None):
        index = len(scenario['steps'])
        prefix = scenario['name'] + '-' + str(index)
        before = {name: value.tobytes() for name, value in inputs.items()}
        bound = {name: bindings.get(name) or dict(file=save(prefix + '-in-' + name, value)) for name, value in inputs.items()}
        row = dict(model=model, inputs=bound)
        started = time.perf_counter()
        try:
            values = sessions[model].run(None, inputs)
        except (Fail, InvalidArgument, RuntimeException) as exc:
            if not failure:
                raise
            row.update(expected_failure='invalid-input', native_error=str(exc), outputs={})
            result = None
        else:
            assert not failure, 'Native unexpectedly accepted invalid input'
            result = {item.name: value for item, value in zip(sessions[model].get_outputs(), values)}
            row['outputs'] = {name: save(prefix + '-out-' + name, value) for name, value in result.items()}
            if repeat is not None:
                row['repeat_of'] = repeat[0]
                assert all(value.dtype == repeat[1][name].dtype and value.shape == repeat[1][name].shape
                           and value.tobytes() == repeat[1][name].tobytes() for name, value in result.items()), 'Native repeat differs'
        assert all(value.tobytes() == before[name] for name, value in inputs.items()), 'Native input changed'
        row['native_seconds'] = time.perf_counter() - started
        scenario['steps'].append(row)
        print(scenario['name'], index, 'rejected' if failure else 'generated', flush=True)
        return result

    def patterned(shape, frequency, scale):
        values = np.arange(np.prod(shape), dtype=np.float64)
        return (np.sin(values * frequency) * scale).astype(np.float32).reshape(shape)

    encoder = dict(name='encoder-lengths', steps=[])
    first_inputs = None
    first_outputs = None
    for batch, frames, lengths in ((1, 64, [64]), (1, 65, [65]), (1, 128, [128]), (1, 256, [256]), (2, 128, [128, 73])):
        inputs = dict(audio_signal=patterned((batch, 128, frames), 0.011, 0.5), length=np.array(lengths, dtype=np.int64))
        outputs = run(encoder, 'encoder', inputs, {})
        if first_inputs is None:
            first_inputs, first_outputs = inputs, outputs
    invalid = dict(audio_signal=np.zeros((1, 127, 64), dtype=np.float32), length=np.array([64], dtype=np.int64))
    run(encoder, 'encoder', invalid, {}, failure=True)
    run(encoder, 'encoder', first_inputs, {}, repeat=(0, first_outputs))
    scenarios.append(encoder)

    for name, batch, frames, tokens, count in (('single', 1, 1, 1, 5), ('multiple', 1, 8, 5, 2), ('batch-two', 2, 3, 2, 2)):
        scenario = dict(name='decoder-' + name, steps=[])
        previous = None
        first_inputs = None
        first_outputs = None
        for index in range(count):
            inputs = dict(encoder_outputs=patterned((batch, 1024, frames), 0.013 + index * 0.001, 1.5),
                          targets=(np.arange(batch * tokens, dtype=np.int32).reshape(batch, tokens) + 1 + index * 7),
                          target_length=np.full((batch,), tokens, dtype=np.int32),
                          input_states_1=np.zeros((2, batch, 640), dtype=np.float32) if previous is None else previous['output_states_1'],
                          input_states_2=np.zeros((2, batch, 640), dtype=np.float32) if previous is None else previous['output_states_2'])
            bindings = {} if previous is None else {f'input_states_{i}':dict(step=index - 1, output=f'output_states_{i}') for i in (1, 2)}
            previous = run(scenario, 'decoder', inputs, bindings)
            if index == 0:
                first_inputs, first_outputs = inputs, previous
        invalid = {name: value.copy() for name, value in first_inputs.items()}
        invalid['targets'].fill(np.iinfo(np.int32).max)
        run(scenario, 'decoder', invalid, {}, failure=True)
        run(scenario, 'decoder', first_inputs, {}, repeat=(0, first_outputs))
        scenarios.append(scenario)

    result = dict(schema=1, scope='parakeet-components', scaled_absolute_tolerance=1e-4, assets=assets,
                  generator_sha256=sha(Path(__file__)), assets_manifest_sha256=sha(assets_path),
                  generator_lf_sha256=source_sha(Path(__file__)), assets_lf_sha256=source_sha(assets_path),
                  numpy=np.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__,
                  native_settings=dict(provider='CPUExecutionProvider', threads=1, execution='sequential', optimization='all', spinning=False),
                  external_data=references, files=files, scenarios=scenarios)
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
