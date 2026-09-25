"""Bind the failed application, qualified products and prior measured graph/counters."""
import hashlib
import json
from pathlib import Path
from source_scope import ORIGINAL, verify

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
MODELS = ROOT / 'artifacts/parakeet-owned-packed-weight-models-amd-20260925'
CENSUS = ROOT / 'artifacts/parakeet-owned-packed-weight-scope-census-amd-20260925'
APP = ROOT / 'artifacts/parakeet-owned-packed-weight-app-amd-20260925'
COUNTERS = ROOT / 'artifacts/parakeet-owned-packed-weight-counters-resume-amd-20260925'
INITIAL = ROOT / 'artifacts/parakeet-owned-packed-weight-counters-v2-amd-20260925'
COST = ROOT / 'artifacts/parakeet-feed-forward-cost-amd-20260925'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def closed(folder, digest):
    assert pin(folder / 'closed.json')['sha256'] == digest
    proof = read(folder / 'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(folder / 'analysis.json')
    for name, wanted in proof['files'].items():
        assert pin(folder / name) == wanted, name
    value = read(folder / 'analysis.json')
    assert value['passed']
    return value


def references():
    models = closed(MODELS, 'bb1a758c6936fa251d921992c67dd7c65ff075fe20829af72f4455b529f1b2ca')
    census = closed(CENSUS, '577a07a60d901cf57c4ba46d70b2c6fc4c02148ab67a1c28fd78d9b4fbcc3d08')
    app = closed(APP, '959d8d180db3431e1aeb5d7ab555fd9b77035ae84d4a8a678d2204cccc71eb48')
    counters = closed(COUNTERS, '1ce66e968f01205e7eb7cee0789cf08f8e74a0f67914f96541f61710253a6db8')
    assert models['no_performance_measurement'] and len(models['results']) == 8
    for name, result in models['results'].items():
        assert result['passed']
        if '-native-' in name:
            assert (result['native']['arrays'], result['native']['values']) == (784, 3090494)
            if name.startswith('candidate-'):
                assert all(r['bit_identical'] for r in result['native']['exact_selected_comparisons'])
        else:
            assert result['public_requests'] == 20
    assert census['product'] == models['identities']['candidate']
    assert counters['products'] == models['identities'] == {'selected': app['identities']['current'], 'candidate': app['identities']['candidate']}
    assert not app['performance']['admitted'] and app['performance']['controls_passed']
    assert [r['name'] for r in app['performance']['gates'] if not r['passed']] == ['corpus-at-least-three-percent-gain']
    assert not counters['application_scored'] and not counters['release_admitted']
    assert counters['model_closure'] == pin(MODELS / 'closed.json') and counters['census_closure'] == pin(CENSUS / 'closed.json')
    source_review = verify(ORIGINAL.read_text(), (TOOLS / 'Program.cs.txt').read_text())
    assert pin(ORIGINAL) == pin(INITIAL / 'build-collected/source/Program.cs')
    initial_receipt = read(INITIAL / 'capture-collected/capture-collection.json')
    assert counters['initial']['collection'] == pin(INITIAL / 'capture-collected/capture-collection.json')
    assert initial_receipt['terminal'] and initial_receipt['code'] == 1
    reference_outputs = {}
    for role, folder in [('selected', INITIAL), ('candidate', COUNTERS)]:
        name = f'probe/{role}-512/result.json'
        path = folder / 'capture-collected' / name
        receipt = read(folder / 'capture-collected/capture-collection.json')
        assert receipt['terminal'] and pin(path) == receipt['files'][name]
        result = read(path)
        assert result['passed'] and result['role'] == role and result['mode'] == '512'
        assert len(result['records']) == 20 and result['immutable_inputs'] and result['held_outputs_independent']
        assert result['core_sha256'] == models['identities'][role]['Lokad.Onnx.dll']['sha256']
        assert result['data_sha256'] == models['identities'][role]['Lokad.Onnx.Data.dll']['sha256']
        reference_outputs[role] = [{k: r[k] for k in ['name', 'frames', 'remainder', 'input_hash', 'frontend', 'encoder', 'copy_bytes', 'scratch_bytes']} for r in result['records']]
    normal, disabled = counters['comparisons']
    assert normal['mode'] == '512' and disabled['mode'] == '256' and normal['clips'] == disabled['clips']
    assert (normal['avoided_packs'], normal['reconstructions']) == (1740, 609)
    graph_path = COST / 'capture-collected/clock/graphs.json'
    published = ROOT / 'tests/parakeet/feed-forward-cost-results/costs-20260925.json'
    assert pin(published)['sha256'] == '233a927b373399a6c110c59a300c84799873d12e3d71eec6c099a09858d88d06'
    cost = read(published)
    assert cost['passed'] and cost['diagnostic_only'] and cost['graph_metadata'] == pin(graph_path)
    reference_path = COST / 'bundle/evidence/cost-reference.json'
    cost_spec = read(COST / 'bundle/spec.json')
    assert pin(COST / 'bundle/spec.json') == read(COST / 'prepared.json')['spec']
    assert pin(reference_path) == cost_spec['files']['evidence/cost-reference.json']
    reference = read(reference_path)
    assert reference['passed'] and reference['graphs'] == read(graph_path)
    graph = reference['graphs']['encoder-model.onnx']
    assert len(graph['nodes']) == 2856 and graph['retained_packed_bytes'] == graph['maximum_packed_bytes'] == 268435456
    nodes = [{k: n[k] for k in ['id', 'name', 'op', 'inputs', 'outputs']} for n in graph['nodes']]
    by_name = {n['name']: n for n in nodes}
    groups = {name: {k: g[k] for k in ['ort_name', 'managed_nodes', 'shape', 'alpha']} for name, g in reference['groups'].items()}
    assert len(groups) == 96 and len(by_name) == 2856
    members = [name for group in groups.values() for name in group['managed_nodes']]
    assert len(members) == len(set(members)) == 144
    for name, group in groups.items():
        assert group['managed_nodes'][0] == name and by_name[name]['op'] == 'MatMul' and '/feed_forward' in name
        assert group['shape'] in [[1024, 4096], [4096, 1024]]
        scale = group['managed_nodes'][1:]
        assert len(scale) == (1 if group['shape'] == [4096, 1024] else 0)
        assert group['alpha'] == (0.5 if scale else 1)
        if scale:
            assert by_name[scale[0]]['op'] == 'Mul' and by_name[name]['outputs'][0] in by_name[scale[0]]['inputs']
    evidence = dict(application_closure=pin(APP / 'closed.json'), counter_closure=pin(COUNTERS / 'closed.json'),
                    source_review=source_review, original_source=pin(ORIGINAL),
                    graph_source=pin(graph_path), group_source=pin(reference_path), graph_manifest=nodes,
                    groups=groups, reference_outputs=reference_outputs, prior_counters=normal['clips'],
                    failed_application_gate='corpus-at-least-three-percent-gain', application_admitted=False,
                    source_inputs={str(p.relative_to(ROOT)): pin(p) for p in [published, graph_path, reference_path, COST / 'prepared.json', COST / 'bundle/spec.json', ORIGINAL]})
    return models, census, evidence
