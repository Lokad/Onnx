"""Bind the admitted complete cases and the explicit short-e5 correction."""
from protocol import pin, read

CASES = ['e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok', 'dinov3', 'resnet50', 'gpt2']


def verify_bundle(base, spec):
    folder = base/'evidence/graph-qualification'
    wanted = spec['graph_qualification']
    assert pin(folder/'closed.json') == wanted['closed']
    assert wanted['closed']['sha256'] == 'ec95b9c7f8019b402fe513b6da6938cf83a8bc2c91f1d5ab65fedd4f7b55ed7c'
    assert pin(folder/'analysis.json') == wanted['analysis']
    proof, value = read(folder/'closed.json'), read(folder/'analysis.json')
    assert proof['passed'] and proof['admitted'] and proof['all_controls_passed']
    assert proof['files']['analysis.json'] == wanted['analysis']
    assert value['passed'] and value['admitted'] and value['all_controls_passed']
    assert not value['root_product_changed'] and value['original_graph_failure_preserved']
    assert (value['clocks'], value['measured'], len(value['setups']), value['source_calls_retained']) == (73512, 8640, 72, 78201)
    assert proof['source_closures'] == value['source_closures']
    assert value['source_closures']['original_graphs']['sha256'] == 'def19d3f178cbb318bc11999b6e23dd18db40772d78949707155cf1f4c791638'
    assert value['source_closures']['short_e5_correction']['sha256'] == 'b81128a6dbea610cf0571279e078debe3ff24c2b0f3c89bbea0e0c4ab675629a'
    for row in value['consumer'].values():
        assert row['passed'] and row['branches_locals_exceptions_equal'] and row['implementation_flags_equal']
        assert not row['product_changed'] and (row['methods'], row['unchanged_methods']) == (66, 65)
    assert [(row['before'], row['after']) for row in value['consumer']['short_e5']['changes']] == [(780, 6180), (600, 6000)]
    assert [row['key'] for row in value['performance']] == CASES
    for row in value['performance']:
        assert row['qualified'] and row['regression_passed'] and row['candidate_over_current'] <= 1.05
        assert [control['role'] for control in row['controls']] == ['current', 'candidate', 'ort']
        assert all(control['passed'] and control['ratio'] <= 1.10 for control in row['controls'])
        assert row['measured_per_process'] == 180
        assert row['warmups'] == (6000 if row['key'] == 'e5-8tok' else 1200 if row['key'] == 'e5-30tok' else 600)
        assert row['source'] == ('short-e5-correction' if row['key'] == 'e5-8tok' else 'original-graphs')
    if 'identities' in spec:
        assert all(value['products'][role]['Lokad.Onnx.dll'] == spec['identities'][label]['Lokad.Onnx.dll']
                   for role, label in [('current', 'selected'), ('candidate', 'candidate')])
    if 'measured' in spec:
        assert value['products']['candidate']['Lokad.Onnx.dll'] == spec['measured']['Lokad.Onnx.dll']
    return dict(passed=True, closed=wanted['closed'], source_closures=value['source_closures'])
