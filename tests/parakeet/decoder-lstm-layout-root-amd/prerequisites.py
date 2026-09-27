"""Require all fresh model/application/graph gates and the exact fixed LSTM layout source."""
from protocol import pin, read
from model_prerequisites import validate as validate_models
from graph_prerequisite import verify_bundle
from application_admission import evaluate

MODEL_LABELS = {'models','parakeet','shared','parakeet-app','baseline'}
LABELS = MODEL_LABELS | {'qualified-root','consumer-qualified','pyannote-app'}


def validate(reports, spec, compatible, contracts, compiled):
    assert spec['source_prepared'] == {'bytes': 308481, 'sha256': 'f0254672a42a78c5a5abb520d10bae67ac38a4e599a059b5cd116e456f11ab78'}
    assert set(reports) == LABELS and all(r['passed'] for r in reports.values())
    validate_models({key: reports[key] for key in MODEL_LABELS}, spec, compatible, reports['consumer-qualified'])
    pair = spec['identities']; selected, candidate = pair['selected'], pair['candidate']
    assert reports['qualified-root']['built'] == selected
    assert reports['qualified-root']['root_source_verified'] and reports['qualified-root']['package']['passed']
    assert compiled == contracts['compiled'] == compatible['compiled']
    assert compiled['passed'] and compiled['public_surface_equal'] and compiled['flags_equal'] and compiled['attributes_equal']
    assert not contracts['contract_regression_found'] and not contracts['original_campaign_passed']
    assert not contracts['performance_measured']
    assert contracts['baseline'] == selected and contracts['candidate'] == candidate
    assert contracts['original_failure'] == {'bytes':463637,'sha256':'59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'}
    assert contracts['matched_existing_scalar_cases'] == 172 and contracts['matched_scalar_rejections'] == 18
    assert contracts['total_projection_values'] == 5836800 and contracts['projection_hashes_equal_across_modes']
    assert set(contracts['modes']) == {'normal','noavx512','scalar'}
    for mode, result in contracts['modes'].items():
        assert (result['passed'],result['failed']) == ((158,18) if mode == 'scalar' else (176,0))
        assert result['added_passed'] == 4 and result['projection_values'] == 1945600 and result['projections'] == 760
        loaded = result['loaded']
        assert loaded['passed'] and loaded['mode'] == mode
        assert loaded['core_sha256'] == candidate['Lokad.Onnx.dll']['sha256']
        assert loaded['avx512'] == (mode == 'normal') and loaded['hardware'] == (mode != 'scalar')
        assert loaded['runtime'] == '10.0.8' and loaded['affinity'] == 4
        assert loaded['block'] == 4*loaded['vector_count']
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
    original = base/'evidence/contracts-failure/failed.json'
    assert pin(original)['sha256'] == '59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'
    failed = read(original); assert not failed['passed'] and failed['terminal'] and failed['evidence_verified']
    folder = base/'evidence/contracts'; proof = read(folder/'closed.json')
    assert pin(folder/'closed.json')['sha256'] == '63c97822999a74921c2e8a3c64e0af9ec6682a6829ba83de52bba353b93b85ec'
    assert proof['passed'] and proof['diagnostic_only'] and not proof['original_campaign_passed']
    assert pin(folder/'analysis.json') == proof['analysis'] == proof['files']['analysis.json']
    contracts = read(folder/'analysis.json'); compiled = contracts['compiled']
    assert pin(base/'evidence/source-prepared.json') == spec['source_prepared']
    assert read(base/'evidence/root-applied.json')['prepared'] == spec['source_prepared']
    validate(reports, spec, read(base/'evidence/product-compatibility.json'), contracts, compiled)
    assert spec['measured'] == spec['identities']['candidate']
    return dict(passed=True, retained=spec['prerequisites'], graph_qualification=graph)
