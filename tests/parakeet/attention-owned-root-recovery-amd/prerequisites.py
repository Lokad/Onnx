"""Require every release gate and the exact attention ownership qualification."""
from protocol import pin,read
from model_prerequisites import validate as validate_models
from graph_prerequisite import verify_bundle
from application_admission import evaluate
from new_cases import NEW_CASES,NEW_SKIPS

MODEL_LABELS={'models','parakeet','shared','parakeet-app','baseline'}
LABELS=MODEL_LABELS|{'qualified-root','consumer-qualified','pyannote-app'}
SOURCE_PIN=dict(bytes=156324,sha256='ee998786951478b241232c66cad94b01e019fcdb9ae644b5b9731e47c3f1ac39')
CONTRACT_PIN=dict(bytes=324767,sha256='a293403a5dc3a7bdabb7fdc448470feea2b54dbf139d98b7b3f9ed0eef39854f')
BUILD_PIN=dict(bytes=4013,sha256='2a1bf8e3c2defbf806dd4aef0f409d6675ff3ea816f79127fd928e09a2e1da64')
CENSUS_PIN=dict(bytes=18794,sha256='ef4e41eca49d33f6cac4ae7d510922053874002787709cc3d83bc98c3437d3d7')


def validate(reports,spec,compatible,contracts,build,census):
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
    assert compatible['focused_contracts']==CONTRACT_PIN and compatible['actual_model_census']==CENSUS_PIN
    assert [(r['mode'],r['passed'],r['skipped']) for r in contracts['suites']]==[('normal',93,0),('256',93,0),('scalar',26,0)]
    normal_classes=dict(MatMulDestinationTests=16,MatMulEmptyTests=4,MatMulVectorTests=3,
        OwnedPackedRuntimeIdentityTests=1,OwnedPackedWeightTests=40,OwnedAttentionPreparationTests=29)
    scalar_classes=dict(MatMulDestinationTests=16,MatMulEmptyTests=4,MatMulVectorTests=3,
        OwnedPackedRuntimeIdentityTests=1,OwnedPackedUnavailableTests=1,OwnedAttentionUnavailableTests=1)
    for row in contracts['suites']:
        assert row['classes']==(scalar_classes if row['mode']=='scalar' else normal_classes)
        assert len(row['names'])==len(set(row['names']))==row['passed']
        added=[n for n in row['names'] if '.OwnedAttention' in n]
        assert set(added)==set(NEW_SKIPS['backend'] if row['mode']=='scalar' else NEW_CASES['backend'])
    assert census['passed'] and census['product']==candidate and census['contracts']==CONTRACT_PIN
    assert census['original_compiled_review']==BUILD_PIN and census['expected_added_attention_weights']==92
    assert not census['release_admitted'] and not census['application_scored']
    assert [r['mode'] for r in census['modes']]==['512','256']
    for mode in census['modes']:
        r=mode['result']
        assert r['passed'] and (r['owned_count'],r['retained_maps'],r['original_weight_count'],r['original_dense_count'])==(179,37,216,37)
        assert r['owned_bytes']==1845493760 and r['retained_clone_bytes']==268435456
        for key in ['public_request_passed','idempotent','identities_preserved','logical_hashes_exact','pcm_unchanged','contexts_share_weights']:
            assert r[key],key
        assert mode['before']['weights']==mode['after']['weights']
        assert mode['before']['maps']==mode['after']['maps']
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
    for name,wanted in [('contracts',CONTRACT_PIN),('census',CENSUS_PIN)]:
        folder=base/'evidence'/name;proof=read(folder/'closed.json')
        assert pin(folder/'closed.json')==wanted and proof['passed']
        assert pin(folder/'analysis.json')==proof['files']['analysis.json']==proof['analysis']
        reviews[name]=read(folder/'analysis.json')
    build=base/'evidence/contracts-build.json';assert pin(build)==BUILD_PIN
    assert pin(base/'evidence/source-prepared.json')==spec['source_prepared']
    applied=read(base/'evidence/root-applied.json')
    assert applied['prepared']==spec['source_prepared']
    assert pin(base/'evidence/OwnedAttentionPreparationTests.cs')==applied['fixture']
    from fixture_repair import repair,verify_source_map
    original=read(base/'evidence/source-prepared.json')['source']
    fixture=(base/'evidence/OwnedAttentionPreparationTests.cs').read_bytes()
    assert fixture==repair((base/'evidence/OwnedAttentionPreparationTests.original.cs.txt').read_bytes())
    verify_source_map(original,applied['source_files'],fixture)
    assert applied['fixture_repair_only'] and applied['recovery_changed']==['tests/Lokad.Onnx.Backend.Tests/OwnedAttentionPreparationTests.cs']
    prior=read(base/'evidence/prior-root-applied.json')
    assert pin(base/'evidence/prior-root-applied.json')==applied['previous_integration']
    assert prior['source_files']==original and prior['prepared']==applied['prepared']
    assert prior['prerequisites']==applied['prerequisites'] and prior['graph_qualification']==applied['graph_qualification']
    failed=read(base/'evidence/failed-root.json')
    assert pin(base/'evidence/failed-root.json')==applied['failed_run']
    assert not failed['passed'] and not failed['release_admitted'] and failed['terminal'] and failed['code']==1
    assert len(applied['source_files'])==446
    validate(reports,spec,read(base/'evidence/product-compatibility.json'),reviews['contracts'],read(build),reviews['census'])
    assert spec['measured']==spec['identities']['candidate']
    return dict(passed=True,retained=spec['prerequisites'],graph_qualification=graph)
