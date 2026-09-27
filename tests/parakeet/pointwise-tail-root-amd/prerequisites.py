"""Require every release gate and preserve the pointwise numerical qualification."""
from protocol import pin, read
from model_prerequisites import validate as validate_models
from graph_prerequisite import verify_bundle
from application_admission import evaluate

MODEL_LABELS = {'models','parakeet','shared','parakeet-app','baseline'}
LABELS = MODEL_LABELS | {'qualified-root','consumer-qualified','pyannote-app'}
SOURCE_PIN = dict(bytes=155332, sha256='9be6d0381e417b435b030b467cabd1d723398ffb9fac294ef8ff91c9b6a6e64c')
ARITHMETIC_PIN = dict(bytes=25881, sha256='558f2a523febd0d794bc7da6cdae3381513b7ff3d75dfd4ca4f133f618821cc8')
BUILD_PIN = dict(bytes=4753, sha256='51090680e6b172287122ef15c5f7e5a3ae2eaa41f083a76cdcb314f418ddc227')
FAILURE_PIN = dict(bytes=289825, sha256='58cb40269af7cda8252007bc059f04ccd4e59697d258e3affbbe48329b8208d9')


def validate(reports, spec, compatible, contracts, build, codegen):
    assert spec['source_prepared'] == SOURCE_PIN
    assert set(reports) == LABELS and all(result['passed'] for result in reports.values())
    validate_models({key: reports[key] for key in MODEL_LABELS}, spec, compatible, reports['consumer-qualified'])
    pair = spec['identities']; selected, candidate = pair['selected'], pair['candidate']
    assert reports['qualified-root']['built'] == selected
    assert reports['qualified-root']['root_source_verified'] and reports['qualified-root']['package']['passed']
    assert build['passed'] and build['source'] == SOURCE_PIN
    assert build['products'] == contracts['products'] == dict(baseline=selected, candidate=candidate)
    assert build['scope'] == compatible['compiled_scope']
    assert contracts['compiled_scope'] == BUILD_PIN
    assert contracts['arithmetic_contract_passed'] and not contracts['all_bits_exact']
    assert contracts['no_performance_measurement'] and not contracts['release_admitted']
    assert contracts['finite_cross_mode_exact'] and contracts['scalar_results_exact']
    assert contracts['original_failed_closure'] == FAILURE_PIN
    assert contracts['baseline_diagnosis'] == dict(bytes=213379, sha256='a199e0920f346ba9c5552f688042c945e9c023a3fd4fa379af4058c6455e5cf0')
    counts = {role+'-'+mode:2598 for role in ['baseline','candidate'] for mode in ['normal','avx512-disabled']}
    counts.update({'scalar-baseline':12, 'scalar-candidate':12})
    assert contracts['cases'] == counts
    assert contracts['failures'] == {name:[] for name in counts}
    payloads = contracts['nan_payload_cases']; assert set(payloads) == set(counts)
    for name, rows in payloads.items():
        count, differences = (1,198) if name == 'candidate-normal' else (274,14097) if name == 'candidate-avx512-disabled' else (0,0)
        assert len(rows) == count and sum(row['differences'] for row in rows) == differences
        assert all(row['exceptional'] and row['oracle'] and row['differences'] > 0 for row in rows)
    assert codegen['passed'] and codegen['numerical_closure'] == ARITHMETIC_PIN
    assert codegen['no_performance_measurement'] and not codegen['release_admitted']
    helpers = codegen['helpers']; assert len(helpers) == 4
    assert {(row['mode'], 'Masked' in row['method']) for row in helpers} == {
        (mode, masked) for mode in ['normal','avx512-disabled'] for masked in [False,True]}
    for row in helpers:
        assert row['unchanged_generated_code'] and row['independent_accumulators'] == 8 and not row['vector_stack_spills']
        assert (row['fma_count'],row['multiply_count'],row['add_count']) == ((0,8,8) if 'Masked' in row['method'] else (8,0,0))
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
    original = base/'evidence/contracts-failure/closed.json'
    assert pin(original) == FAILURE_PIN
    failed = read(original); assert failed['completed'] and not failed['passed']
    folder = base/'evidence/contracts'; proof = read(folder/'closed.json')
    assert pin(folder/'closed.json') == ARITHMETIC_PIN
    assert proof['completed'] and proof['arithmetic_contract_passed']
    assert pin(folder/'analysis.json') == proof['files']['analysis.json']
    build = base/'evidence/contracts-build.json'; codegen = base/'evidence/contracts-codegen.json'
    assert pin(build) == BUILD_PIN
    assert pin(codegen)['sha256'] == '3715408c95a3164b2f06e36db4aeb2bea95f08ee8f7afb03cbb38e9747d9d692'
    assert pin(base/'evidence/source-prepared.json') == spec['source_prepared']
    applied = read(base/'evidence/root-applied.json')
    assert applied['prepared'] == spec['source_prepared']
    assert pin(base/'evidence/PackedColumnRemainderTests.cs') == applied['fixture']
    validate(reports, spec, read(base/'evidence/product-compatibility.json'), read(folder/'analysis.json'), read(build), read(codegen))
    assert spec['measured'] == spec['identities']['candidate']
    return dict(passed=True, retained=spec['prerequisites'], graph_qualification=graph)
