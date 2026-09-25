"""Verify and reuse the exact retained graph, routes and native correspondence."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def references():
    feed_path = ROOT / 'tests/parakeet/slice-dense-conversion-results/feed-forward-review-20260925.json'
    gap_path = ROOT / 'tests/parakeet/slice-dense-conversion-results/remaining-gap-20260925.json'
    routes_path = ROOT / 'artifacts/parakeet-projection-route-resume-amd-20260924/matched-projection-calls.json'
    graphs_path = ROOT / 'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924/capture-collected/wall/graphs.json'
    observer_path = ROOT / 'tests/parakeet/slice-dense-conversion-results/observer-20260925.json'
    profile = ROOT / 'artifacts/parakeet-slice-dense-conversion-profile-resume-amd-20260925'
    feed, gap, observer = read(feed_path), read(gap_path), read(observer_path)
    assert feed['passed'] and gap['passed'] and observer['passed']
    assert feed['current_candidate_profile'] == pin(gap_path)
    assert feed['earlier_matched_calls'] == pin(routes_path)
    assert observer['graph_metadata'] == pin(graphs_path)
    proof = read(profile / 'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(profile / 'analysis.json')
    assert gap['candidate_profile'] == dict(closure=pin(profile / 'closed.json'), analysis=pin(profile / 'analysis.json'))
    assert pin(profile / 'closed.json')['sha256'] == '705fad8054d2f8020e99e80b32dca43f8f8c6ab046a63780a65faa062a427990'
    graphs = read(graphs_path)
    original = {n['name']: n for n in graphs['encoder-model.onnx']['nodes']}
    rows = [n for n in read(profile / 'analysis.json')['phases']['candidate']['node_rows'] if n['graph'] == 'encoder']
    assert len(rows) == len(original) == 2856
    for row in rows:
        assert all(row[key] == original[row['name']][key] for key in ['id', 'op', 'inputs', 'outputs', 'constant_inputs'])
    groups = {g['managed_nodes'][0]: g for g in gap['projections'] if '/feed_forward' in g['managed_nodes'][0]}
    assert len(groups) == 96
    assert Counter((tuple(g['shape']), g['alpha']) for g in groups.values()) == Counter({((1024, 4096), 1.): 48, ((4096, 1024), .5): 48})
    assert len({n for g in groups.values() for n in g['managed_nodes']}) == 144
    routes = [r for r in read(routes_path) if r['name'] in groups]
    assert len(routes) == len({(r['request'], r['name']) for r in routes}) == 7680
    assert all(r['copy_bytes'] == 0 and r['frames'] == r['m'] and not r['positional'] for r in routes)
    assert Counter(r['route'] for r in routes) == Counter({'mapped-allowed': 468, 'mapped-declined': 252, 'unmapped': 6960})
    files = [feed_path, gap_path, routes_path, graphs_path, observer_path, profile / 'closed.json', profile / 'analysis.json']
    return dict(passed=True, graphs=graphs, routes=routes, groups=groups,
        numeric_ops={r['name']: r['numeric_op'] for r in rows},
        inputs={p.relative_to(ROOT).as_posix(): pin(p) for p in files},
        native_profile_is_earlier=True, per_node_copy_counters_are_retained_not_new=True)


if __name__ == '__main__':
    value = references()
    print(json.dumps(dict(passed=value['passed'], encoder_nodes=len(value['numeric_ops']),
                          groups=len(value['groups']), matched_calls=len(value['routes']), input_files=len(value['inputs']))))
