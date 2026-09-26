"""All fresh application and graph admission precedes the actual root build."""
from protocol import pin,read
from model_prerequisites import validate as validate_models
from graph_prerequisite import verify_bundle
from application_admission import evaluate

MODEL_LABELS={'models','parakeet','shared','parakeet-app','baseline'}
LABELS=MODEL_LABELS|{'build','qualified-root','pyannote-app'}


def validate(reports,spec):
    assert set(reports)==LABELS and all(r['passed'] for r in reports.values())
    validate_models({key:reports[key] for key in MODEL_LABELS},spec)
    pair=spec['identities'];selected,candidate=pair['selected'],pair['candidate']
    build=reports['build'];assert build['measured']==selected and build['built']==candidate
    assert build['source_prepared']==spec['source_prepared']
    proof=build['inventory']
    assert proof['passed'] and proof['core_existing_methods']==3281 and proof['core_unchanged_methods']==3280
    assert proof['data_unchanged_methods']==697 and proof['only_public_pad_changed']
    assert proof['original_padcore_exact'] and proof['public_surface_equal'] and proof['existing_implementation_flags_equal']
    assert proof['generated_names_exact'] and proof['helper_matches_reviewed_original']
    assert proof['public_pad']['call_targets_changed']==4 and proof['public_pad']['branches_locals_exceptions_equal']
    assert all(r['census_exact'] and (r['passed'],r['skipped'])==(6,0) for r in build['suites'].values())
    assert set(build['suites'])=={'pad-tests','pad-tests-256'}
    assert reports['qualified-root']['built']==selected
    app=reports['pyannote-app']
    assert app['identities']==pair and app['consumers']==spec['consumers']
    assert app['performance']==evaluate(app['table']) and app['performance']['admitted']
    assert len(app['performance']['controls'])==12 and len(app['performance']['gates'])==4
    assert (app['timing_requests'],app['warmup'],app['measured'])==(96,24,72)
    assert (app['native_public_requests'],app['meeting_requests'])==(4,3)
    assert app['reference_provenance_verified']
    expected=['native-pyannote','meetings-inputs','meetings-run']+[
        f'timing-{i:02}-{role}' for i,role in enumerate(['selected','candidate','ort','ort','candidate','selected'])]
    assert list(app['results'])==expected and all(r['passed'] for r in app['results'].values())
    meetings=app['results']['meetings-run']
    assert meetings['calls']==3 and meetings['complete_selected_results_exact']
    assert len(meetings['comparisons'])==6 and all(r['passed'] for r in meetings['comparisons'])
    return True


def verify(base,spec):
    graph=verify_bundle(base,spec);reports={}
    assert set(spec['prerequisites'])==LABELS
    for name,wanted in spec['prerequisites'].items():
        folder=base/'evidence'/name
        assert pin(folder/'closed.json')==wanted['closed']
        proof=read(folder/'closed.json')
        assert proof['passed'] and proof['analysis']==wanted['analysis']==pin(folder/'analysis.json')
        if name in ['parakeet-app','pyannote-app']:assert proof['admitted']
        reports[name]=read(folder/'analysis.json')
    validate(reports,spec)
    assert spec['measured']==spec['identities']['candidate']
    return dict(passed=True,retained=spec['prerequisites'],graph_qualification=graph)
