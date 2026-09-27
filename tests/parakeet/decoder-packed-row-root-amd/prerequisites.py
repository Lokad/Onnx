"""Require all fresh model/application/graph gates and the exact fixed prepared-row source."""
from protocol import pin, read
from model_prerequisites import validate as validate_models
from graph_prerequisite import verify_bundle
from application_admission import evaluate

MODEL_LABELS = {'models','parakeet','shared','parakeet-app','baseline'}
LABELS = MODEL_LABELS | {'qualified-root','consumer-qualified','pyannote-app'}


def validate(reports, spec, compatible, contracts, compiled):
    assert spec['source_prepared'] == {'bytes': 163918, 'sha256': '0d72754366d174c3915ffdb07e8a0acf81c9d07dc1e52a9b0286e3135b4fabe1'}
    assert set(reports) == LABELS and all(r['passed'] for r in reports.values())
    validate_models({key: reports[key] for key in MODEL_LABELS}, spec, compatible, reports['consumer-qualified'])
    pair = spec['identities']; selected, candidate = pair['selected'], pair['candidate']
    assert reports['qualified-root']['built'] == selected
    assert reports['qualified-root']['root_source_verified'] and reports['qualified-root']['package']['passed']
    assert compiled == contracts['compiled'] == compatible['compiled']
    assert compiled['passed'] and compiled['public_surface_equal'] and compiled['flags_equal'] and compiled['assembly_metadata_equal']
    assert contracts['passed'] and contracts['root_product_changed'] is False
    assert contracts['products'] == dict(current=selected['Lokad.Onnx.dll'], candidate=candidate['Lokad.Onnx.dll'])
    assert set(contracts['contracts']) == {'normal','noavx512','scalar'}
    for mode, roles in contracts['contracts'].items():
        assert set(roles) == {'current','candidate'}
        for role, result in roles.items():
            assert result['passed'] and result['mode'] == mode and result['role'] == role
            assert result['core_sha256'] == contracts['products'][role]['sha256']
            assert len(result['public_cases']) == 45
            assert sum(r['checked_values'] for r in result['public_cases']) == 828453
            assert all(r['exact'] and r['owned_outputs'] and r['immutable_inputs'] and r['copies'] == r['scratches'] == 0 for r in result['public_cases'])
            raw = result['raw_cases']; count = 81 if role == 'candidate' and mode != 'scalar' else 0
            assert len(raw) == count
            assert all(r['exact'] and r['guards'] and r['immutable_inputs'] and r['allocated_bytes_for_eight_calls'] == 0 for r in raw)
            if count: assert sum(r['checked_values'] for r in raw) == 109030
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
    assert pin(folder/'closed.json')['sha256'] == 'fc00688c8ef0e65e5d5109f1808209add023e740aabc5b4312647e547df7d61d'
    assert proof['passed']
    assert pin(folder/'analysis.json') == proof['analysis'] == proof['files']['analysis.json']
    contracts = read(folder/'analysis.json'); compiled = contracts['compiled']
    assert pin(base/'evidence/source-prepared.json') == spec['source_prepared']
    assert read(base/'evidence/root-applied.json')['prepared'] == spec['source_prepared']
    validate(reports, spec, read(base/'evidence/product-compatibility.json'), contracts, compiled)
    assert spec['measured'] == spec['identities']['candidate']
    return dict(passed=True, retained=spec['prerequisites'], graph_qualification=graph)
