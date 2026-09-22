"""Inspect the local ORT optimized Pyannote graph without executing inference."""
import collections
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/pyannote-nchwc-windows-20260922'
SITE = ROOT / 'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
MODELS = dict(embedding=ROOT / 'models/pyannote-embedding/embedding_encoder.onnx',
              segmentation=ROOT / 'models/pyannote-segmentation/segmentation/model.onnx')
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('graph_census_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify = monitor.pin, monitor.read, monitor.save, monitor.verify


def child():
    import onnx
    import onnxruntime as ort
    import numpy as np
    assert ort.__version__ == '1.29.0' and onnx.__version__ == '1.22.0' and np.__version__ == '2.2.4'
    assert monitor.psutil.Process().cpu_affinity() == [2]
    def describe(path):
        model = onnx.load(str(path), load_external_data=False)
        weights = {v.name: v for v in model.graph.initializer}
        nodes = []
        for node in model.graph.node:
            if node.op_type in ['Conv', 'FusedConv', 'ReorderInput', 'ReorderOutput']:
                attrs = {}
                for a in node.attribute:
                    if a.type == onnx.AttributeProto.INT: attrs[a.name] = a.i
                    elif a.type == onnx.AttributeProto.INTS: attrs[a.name] = list(a.ints)
                    elif a.type == onnx.AttributeProto.STRING: attrs[a.name] = a.s.decode()
                    elif a.type == onnx.AttributeProto.FLOAT: attrs[a.name] = a.f
                nodes.append(dict(name=node.name, domain=node.domain, op=node.op_type, inputs=list(node.input),
                    outputs=list(node.output), attributes=attrs,
                    weight_shape=list(weights[node.input[1]].dims) if len(node.input) > 1 and node.input[1] in weights else None))
        counts = collections.Counter((n.domain, n.op_type) for n in model.graph.node)
        return dict(file=pin(path), nodes=len(model.graph.node),
            counts=[dict(domain=d, op=o, count=c) for (d, o), c in sorted(counts.items())], convolution_and_reorders=nodes)
    results = {}
    for name, model in MODELS.items():
        target = BASE / (name + '-optimized.onnx')
        options = ort.SessionOptions(); options.log_severity_level = 4
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ['session.intra_op.allow_spinning', 'session.inter_op.allow_spinning']: options.add_session_config_entry(key, '0')
        options.optimized_model_filepath = str(target)
        session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        assert session.get_providers() == ['CPUExecutionProvider'] and target.is_file()
        results[name] = dict(original=describe(model), optimized=describe(target),
            inputs=[dict(name=v.name, shape=v.shape, type=v.type) for v in session.get_inputs()])
        del session
    save(BASE / 'result.json', dict(passed=True, inference_executed=False, platform=sys.platform,
        ort_version=ort.__version__, onnx_version=onnx.__version__, numpy_version=np.__version__,
        affinity=[2], providers=['CPUExecutionProvider'], intra_threads=1, inter_threads=1,
        sequential=True, graph_optimizations='all', spinning=False, models=results))


def run():
    assert not BASE.exists(); BASE.mkdir(); (BASE / 'logs').mkdir()
    dependencies = [Path(__file__).resolve(), MONITOR, *MODELS.values(),
        ROOT / 'artifacts/ort-nchwc-source-20260922/source.json',
        ROOT / 'artifacts/ort-nchwc-source-20260922/nchwc_transformer.cc',
        ROOT / 'artifacts/ort-nchwc-source-20260922/nchwc_ops.cc',
        ROOT / 'artifacts/ort-nchwc-execution-source-20260922/source.json',
        ROOT / 'artifacts/ort-nchwc-execution-source-20260922/snchwc.cpp']
    dependencies.extend(p for package in ['onnxruntime', 'onnx', 'numpy'] for p in (SITE / package).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts)
    files = {p.relative_to(ROOT).as_posix(): pin(p) for p in dependencies}
    save(BASE / 'inputs.json', dict(passed=True, files=files))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None,
        supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    try:
        monitor.worker(state, BASE / 'processes.json', 'metadata', [sys.executable, '-X', 'utf8', '-B',
            Path(__file__).resolve(), 'child'], ROOT, [0], 8, 8, 900, False, BASE)
        verify(files); assert read(BASE / 'result.json')['inference_executed'] is False
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'processes.json', state)


def audit():
    assert not (BASE / 'closed.json').exists()
    spec = importlib.util.spec_from_file_location('graph_census_audit', ROOT / 'tests/pyannote/convolution-portable-qualification/common.py')
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    files = read(BASE / 'inputs.json')['files']; verify(files)
    resources = common.resources(BASE, 'processes.json', dict(metadata=(8, 900, True)))
    result = read(BASE / 'result.json'); assert result['passed'] and not result['inference_executed']
    assert result['platform'] == 'win32' and set(result['models']) == set(MODELS)
    for name, value in result['models'].items():
        assert value['original']['file'] == pin(MODELS[name])
        assert value['optimized']['file'] == pin(BASE / (name + '-optimized.onnx'))
        for kind in ['original', 'optimized']:
            assert sum(r['count'] for r in value[kind]['counts']) == value[kind]['nodes']
    common.close(BASE, dict(passed=True, result=result, scope='Windows graph construction only; no model inference, AMD dispatch or performance claim', **resources), files, resources['identities'])


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['run', 'child', 'audit']
    globals()[sys.argv[1]]()
