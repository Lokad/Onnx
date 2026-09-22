"""Read-only graph and output inspection for the selected AMD native diagnostic."""
import collections
import hashlib
import json
import math
from pathlib import Path

GIB = 1024**3
LIMITS = dict(preflight_available=12*GIB, preflight_tmpfs=3*GIB, rss=8*GIB,
              available=GIB, tmpfs=GIB, output=GIB, artifacts=2*GIB, seconds=900)
MODELS = ['embedding', 'segmentation']


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def verify(base):
    spec = read(base/'payload.json'); assert spec['limits'] == LIMITS
    for name, wanted in spec['files'].items():
        path = (base/name).resolve(); assert path.is_relative_to(base.resolve())
        assert pin(path) == wanted, name
    for name, wanted in spec['external'].items(): assert pin(name) == wanted, name
    return spec


def graph(path):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    rows = []
    def visit(g, scope):
        weights = {v.name: v for v in g.initializer}
        for index, node in enumerate(g.node):
            attrs = {}; subgraphs = []
            for a in node.attribute:
                if a.type == onnx.AttributeProto.INT: attrs[a.name] = a.i
                elif a.type == onnx.AttributeProto.INTS: attrs[a.name] = list(a.ints)
                elif a.type == onnx.AttributeProto.FLOAT: attrs[a.name] = a.f
                elif a.type == onnx.AttributeProto.FLOATS: attrs[a.name] = list(a.floats)
                elif a.type == onnx.AttributeProto.STRING: attrs[a.name] = a.s.decode()
                elif a.type == onnx.AttributeProto.GRAPH: subgraphs.append((a.name, a.g))
                elif a.type == onnx.AttributeProto.GRAPHS: subgraphs.extend((a.name+'/'+str(i), v) for i,v in enumerate(a.graphs))
            rows.append(dict(scope=scope, index=index, name=node.name, domain=node.domain, op=node.op_type,
                inputs=list(node.input), outputs=list(node.output), attributes=attrs,
                weights={n: dict(shape=list(weights[n].dims), dtype=weights[n].data_type) for n in node.input if n in weights}))
            for name, nested in subgraphs: visit(nested, scope+'/'+str(index)+'/'+name)
    visit(model.graph, 'main')
    counts = collections.Counter((r['domain'], r['op']) for r in rows if r['scope'] == 'main')
    return dict(file=pin(path), top_level_nodes=len(model.graph.node), nodes=rows,
                counts=[dict(domain=d, op=o, count=c) for (d,o),c in sorted(counts.items())])


def compare(actual, expected):
    import numpy as np
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))/np.maximum(1., np.abs(expected.astype(np.float64)))
    return dict(values=int(actual.size), failed_values=int(np.count_nonzero(delta > 1e-4)),
                maximum=float(delta.max(initial=0)), identical=actual.tobytes() == expected.tobytes())


def profile_events(events, census):
    by_name = collections.defaultdict(list)
    for row in census['nodes']: by_name[row['name']].append(row)
    kernels = []
    for index, event in enumerate(events):
        if event.get('cat') != 'Node' or not event.get('name', '').endswith('_kernel_time'): continue
        name = event['name'][:-len('_kernel_time')]
        matches = by_name[name]
        assert len(matches) == 1, ('Unresolved executed node', name, len(matches))
        node = matches[0]; args = event['args']
        assert args['op_name'] == node['op'] and args['provider'] == 'CPUExecutionProvider'
        assert isinstance(event['dur'], (int,float)) and math.isfinite(event['dur']) and event['dur'] >= 0
        kernels.append(dict(event_index=index, name=name, scope=node['scope'], domain=node['domain'],
                            op=node['op'], duration_us=event['dur']))
    assert kernels
    # Six runs must execute every top-level node. Branch interiors are retained
    # and resolved separately; an untaken branch need not have profile events.
    observed = collections.Counter((r['scope'],r['name']) for r in kernels)
    for node in census['nodes']:
        if node['scope'] == 'main': assert observed[('main',node['name'])] == 6, node['name']
    runs = [e for e in events if e.get('name') == 'model_run' and e.get('cat') == 'Session']
    assert len(runs) == 6
    return dict(events=len(events), model_runs=len(runs), kernels=kernels)


def check_sample(row):
    assert 0 <= row['seconds'] < LIMITS['seconds']
    assert row['rss'] == sum(m['rss'] for m in row['members']) < LIMITS['rss']
    assert row['available'] >= LIMITS['available'] and row['tmpfs'] >= LIMITS['tmpfs']
    assert row['output'] <= LIMITS['output'] and row['artifacts'] <= LIMITS['artifacts']
    for m in row['members']:
        assert m['affinity'] == [2] and m['threads']
        assert all(t['affinity'] == [2] for t in m['threads'])
