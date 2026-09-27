"""Require all fresh model/application/graph gates and the exact fixed arithmetic source."""
from protocol import pin, read
from model_prerequisites import validate as validate_models
from graph_prerequisite import verify_bundle
from application_admission import evaluate

MODEL_LABELS = {'models','parakeet','shared','parakeet-app','baseline'}
LABELS = MODEL_LABELS | {'qualified-root','consumer-qualified','pyannote-app'}


def validate(reports, spec, compatible, contracts, compiled):
    assert set(reports) == LABELS and all(r['passed'] for r in reports.values())
    validate_models({key: reports[key] for key in MODEL_LABELS}, spec, compatible, reports['consumer-qualified'])
    pair = spec['identities']; selected, candidate = pair['selected'], pair['candidate']
    assert reports['qualified-root']['built'] == selected
    assert reports['qualified-root']['root_source_verified'] and reports['qualified-root']['package']['passed']
    assert compiled['passed'] and compiled['product'] == candidate and compiled['source'] == spec['source_prepared']
    assert compiled['public_surface_unchanged'] and compiled['data_methods_unchanged'] and compiled['shared_exp_unchanged']
    assert compiled['zero_added_warnings'] and not compiled['release_admitted']
    core, data = compiled['methods']
    assert (core['assembly'], core['methods'], core['unchanged']) == ('Lokad.Onnx.dll',3282,3281)
    assert len(core['changed']) == len(core['added']) == 1 and not core['removed']
    assert '::Sigmoid::' in core['changed'][0] and '::SigmoidRationalVector::' in core['added'][0]
    assert (data['assembly'], data['methods'], data['unchanged']) == ('Lokad.Onnx.Data.dll',697,697)
    assert not data['changed'] and not data['added'] and not data['removed']
    assert contracts['passed'] and contracts['product'] == candidate and contracts['source'] == spec['source_prepared']
    assert contracts['compiled_review']['sha256'] == '2619dee6c8f64ea91578d8c63665878f8ed1181da8cbf464dd0514b4af2092e7'
    assert [(s['mode'],s['passed'],s['skipped']) for s in contracts['suites']] == [('normal',12,0),('scalar',12,0)]
    for row in contracts['suites']:
        sweep = row['sweep']
        assert sweep['passed'] and sweep['checked_values'] == 2048769 and sweep['tolerance'] == 1e-6
        assert 0 <= sweep['maximum_absolute_error'] <= 1e-6
    app = reports['pyannote-app']
    assert app['identities'] == pair and app['consumers'] == spec['consumers']
    assert app['performance'] == evaluate(app['table']) and app['performance']['admitted']
    assert len(app['performance']['controls']) == 12 and len(app['performance']['gates']) == 4
    assert (app['timing_requests'],app['warmup'],app['measured']) == (96,24,72)
    assert (app['native_public_requests'],app['meeting_requests']) == (4,3)
    assert app['reference_provenance_verified']
    expected = ['native-pyannote','meetings-inputs','meetings-run'] + [
        f'timing-{i:02}-{role}' for i, role in enumerate(['selected','candidate','ort','ort','candidate','selected'])]
    assert list(app['results']) == expected and all(r['passed'] for r in app['results'].values())
    meetings = app['results']['meetings-run']
    assert meetings['calls'] == 3 and meetings['complete_selected_results_exact']
    assert len(meetings['comparisons']) == 6 and all(r['passed'] for r in meetings['comparisons'])
    return True


def verify(base, spec):
    graph = verify_bundle(base, spec); reports = {}
    assert set(spec['prerequisites']) == LABELS
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name
        assert pin(folder/'closed.json') == wanted['closed']
        proof = read(folder/'closed.json')
        assert proof['passed'] and proof['analysis'] == wanted['analysis'] == pin(folder/'analysis.json')
        if name in ['parakeet-app','pyannote-app','consumer-qualified']: assert proof['admitted']
        reports[name] = read(folder/'analysis.json')
    folder = base/'evidence/contracts'; proof = read(folder/'closed.json')
    assert pin(folder/'closed.json')['sha256'] == '3301ae58b42e1fd51435f54191cf4bba42c6cbf8e2ad6bf4279af67ae5d3d89a'
    assert proof['passed']
    for name in ['analysis.json','build-review.json']: assert pin(folder/name) == proof['files'][name]
    contracts, compiled = read(folder/'analysis.json'), read(folder/'build-review.json')
    assert contracts['compiled_review'] == pin(folder/'build-review.json')
    validate(reports, spec, read(base/'evidence/product-compatibility.json'), contracts, compiled)
    assert spec['measured'] == spec['identities']['candidate']
    return dict(passed=True, retained=spec['prerequisites'], graph_qualification=graph)
