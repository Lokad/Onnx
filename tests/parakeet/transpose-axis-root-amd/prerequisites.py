"""Require every release gate and the exact transpose dispatch qualification."""
from protocol import pin,read
from model_prerequisites import validate as validate_models
from graph_prerequisite import verify_bundle
from application_admission import evaluate
from new_cases import NEW_CASES,NEW_SKIPS

MODEL_LABELS={'models','parakeet','shared','parakeet-app','baseline'}
LABELS=MODEL_LABELS|{'qualified-root','consumer-qualified','pyannote-app'}
SOURCE_PIN=dict(bytes=156606,sha256='d311c9d133a89a1bcf341dfff64e6e6cf8cb75057f8c93bc94f10350aa79408e')
CONTRACT_PIN=dict(bytes=332142,sha256='c0c063bb3f73eed2d8d7372d17dd1da3ce4f6a962e190ef101b504fb245e4ecd')
BUILD_PIN=dict(bytes=4378,sha256='f41cc8c3658a43c8348150f88df123691d6fdcee806a2844949aa3994c790d45')


def validate(reports,spec,compatible,contracts,build):
    assert spec['source_prepared']==SOURCE_PIN
    assert set(reports)==LABELS and all(result['passed'] for result in reports.values())
    validate_models({key:reports[key] for key in MODEL_LABELS},spec,compatible,reports['consumer-qualified'])
    pair=spec['identities'];selected,candidate=pair['selected'],pair['candidate']
    assert reports['qualified-root']['built']==selected
    assert reports['qualified-root']['root_source_verified'] and reports['qualified-root']['package']['passed']
    assert build['passed'] and build['source']==contracts['source']==SOURCE_PIN
    assert build['product']==contracts['product']==candidate
    assert build['methods']==compatible['compiled_scope']
    assert build['arithmetic_leaves_unchanged'] and build['data_binary_unchanged']
    assert selected['Lokad.Onnx.Data.dll']==candidate['Lokad.Onnx.Data.dll']
    assert contracts['passed'] and contracts['compiled_review']==BUILD_PIN
    assert not contracts['release_admitted'] and contracts['no_application_score']
    assert compatible['focused_contracts']==CONTRACT_PIN
    expected=[(mode,suite,38 if suite=='backend' else 83 if mode=='scalar' else 84,0)
              for mode in ['normal','256','scalar','faces-off'] for suite in ['backend','tensors']]
    assert [(r['mode'],r['suite'],r['passed'],r['skipped']) for r in contracts['suites']]==expected
    backend=dict(TransposeBlockAgreementTests=18,TransposeBoundaryTests=9,TransposeFastPathTests=4,
                 TransposeAxisMovementTests=6,OwnedPackedRuntimeIdentityTests=1)
    tensors=dict(TransposeFaceDispatchTests=76,TensorTransposeFaceTests=1,TensorTranspose8x8Tests=1,
                 TensorReshapeTransposeTests=2,NoOptionalParametersTests=4)
    for row in contracts['suites']:
        wanted=dict(backend if row['suite']=='backend' else tensors)
        if row['suite']=='tensors' and row['mode']=='scalar':wanted.pop('TensorTranspose8x8Tests')
        assert row['classes']==wanted
        assert len(row['names'])==len(set(row['names']))==row['passed']
        added=[n for n in row['names'] if '.TransposeAxisMovementTests.' in n]
        assert set(added)==set(NEW_CASES[row['suite']])
    assert sum(row['passed'] for row in contracts['suites'])==487
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


def verify(base,spec):
    graph=verify_bundle(base,spec);reports={}
    assert set(spec['prerequisites'])==LABELS
    for name,wanted in spec['prerequisites'].items():
        folder=base/'evidence'/name
        assert pin(folder/'closed.json')==wanted['closed']
        proof=read(folder/'closed.json')
        assert proof['passed'] and proof['analysis']==wanted['analysis']==pin(folder/'analysis.json')
        if name in ['parakeet-app','pyannote-app','consumer-qualified']:assert proof['admitted']
        reports[name]=read(folder/'analysis.json')
    reviews={}
    for name,wanted in [('contracts',CONTRACT_PIN)]:
        folder=base/'evidence'/name;proof=read(folder/'closed.json')
        assert pin(folder/'closed.json')==wanted and proof['passed']
        assert pin(folder/'analysis.json')==proof['files']['analysis.json']==proof['analysis']
        reviews[name]=read(folder/'analysis.json')
    build=base/'evidence/contracts-build.json';assert pin(build)==BUILD_PIN
    assert pin(base/'evidence/source-prepared.json')==spec['source_prepared']
    applied=read(base/'evidence/root-applied.json')
    assert applied['prepared']==spec['source_prepared']
    assert pin(base/'evidence/TransposeAxisMovementTests.cs')==applied['fixture']
    assert applied['source_files']==read(base/'evidence/source-prepared.json')['source']
    assert len(applied['source_files'])==447
    validate(reports,spec,read(base/'evidence/product-compatibility.json'),reviews['contracts'],read(build))
    assert spec['measured']==spec['identities']['candidate']
    return dict(passed=True,retained=spec['prerequisites'],graph_qualification=graph)
