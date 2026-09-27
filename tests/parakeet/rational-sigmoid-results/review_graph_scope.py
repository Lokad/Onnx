"""Read the exact regression exports; establish whether they contain Sigmoid."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import onnx

ROOT = Path(__file__).resolve().parents[3]
GRAPH = ROOT/'artifacts/parakeet-pad-current-graphs-v2-amd-20260926'
MODELS = ROOT/'artifacts/parakeet-rational-sigmoid-models-amd-20260927'
OUTPUT = Path(__file__).with_name('graph-scope-20260927.json')


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def review():
    inputs = {}
    for folder, digest in [(GRAPH, 'bea082c63f1e4497f2eacfc3e6bd77325e63b4367c6ce40e3fb3fe69964cb96f'),
                           (MODELS, '3d2e01b5aca3f66c434d5d1e8124c2b0d08327bc14dd37409c0e1884619769d8')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        assert read(folder/'closed.json')['passed']
        inputs[(folder/'closed.json').relative_to(ROOT).as_posix()] = pin(folder/'closed.json')
    proof = read(GRAPH/'closed.json')
    for name in ['payload.json', 'collected/cases.json']:
        assert pin(GRAPH/name) == proof['files'][name]
        inputs[(GRAPH/name).relative_to(ROOT).as_posix()] = pin(GRAPH/name)
    compatible = MODELS/'collected/evidence/compatibility.json'
    assert pin(compatible) == read(MODELS/'closed.json')['files']['collected/evidence/compatibility.json']
    inputs[compatible.relative_to(ROOT).as_posix()] = pin(compatible)
    change = read(compatible)
    assert change['passed'] and change['changed_core_methods'] == ['Sigmoid']
    assert change['added_private_methods'] == ['SigmoidRationalVector'] and change['underlying_methods_reconciled'] == 3979
    cases, spec = read(GRAPH/'collected/cases.json')['cases'], read(GRAPH/'payload.json')
    assert len(cases) == 8
    rows = []
    for name in sorted({r['model'] for r in cases}):
        path = ROOT/name.removeprefix('/home/vermorel/Onnx/')
        assert path.is_relative_to(ROOT/'models') and pin(path) == spec['external'][name]
        model = onnx.load_model(path, load_external_data=False)
        assert len(model.functions) == 0
        counts = collections.Counter(); graphs = [model.graph]; graph_count = 0
        while graphs:
            graph = graphs.pop(); graph_count += 1
            for node in graph.node:
                assert node.domain in ['', 'ai.onnx'], (name, node.domain)
                counts[node.op_type] += 1
                for attribute in node.attribute:
                    if attribute.type == onnx.AttributeProto.GRAPH: graphs.append(attribute.g)
                    elif attribute.type == onnx.AttributeProto.GRAPHS: graphs.extend(attribute.graphs)
        assert counts['Sigmoid'] == 0
        rows.append(dict(model=path.relative_to(ROOT).as_posix(), identity=pin(path),
            cases=[r['key'] for r in cases if r['model'] == name], graphs=graph_count,
            functions=0, nodes=sum(counts.values()), operators=dict(sorted(counts.items())), sigmoid_nodes=0))
    assert len(rows) == 4
    return dict(passed=True, inputs=inputs, models=rows, onnx_parser_version=onnx.__version__,
        inference_calls=0, graph_transforms_executed=False, runtime_dispatch_measured=False,
        conclusion='All four exact regression exports, including nested graphs, contain no Sigmoid. Keep the original candidate/current byte-equality check for the fresh graph regression; this static census is not numerical or performance qualification.')


if __name__ == '__main__':
    assert sys.argv[1:] in [[], ['--publish']]
    value = review()
    if sys.argv[1:]:
        with OUTPUT.open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    else: assert read(OUTPUT) == value
    print(json.dumps(dict(passed=True, models=[dict(model=r['model'], nodes=r['nodes'], sigmoid=r['sigmoid_nodes']) for r in value['models']], inference_calls=0)))
