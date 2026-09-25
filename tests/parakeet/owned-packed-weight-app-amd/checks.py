"""Verify current product prerequisites and every complete Parakeet timing request."""
import importlib.util
from fractions import Fraction
from protocol import pin, read


def records_protocol(base):
    spec = importlib.util.spec_from_file_location('original_audio_records', base/'runtime/protocol.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prereqs(base, spec):
    assert set(spec['prerequisites'])=={'baseline','models','contracts','census','counters','selected_app'}
    reports={}
    for name,wanted in spec['prerequisites'].items():
        folder=base/'evidence'/name;proof=read(folder/'closed.json')
        assert pin(folder/'closed.json')==wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json')==wanted['analysis']==proof['analysis']
        reports[name]=read(folder/'analysis.json');assert reports[name]['passed']
    current=spec['identities']['current'];candidate=spec['identities']['candidate']
    baseline,models,contracts,census,counters,selected=[reports[n] for n in ['baseline','models','contracts','census','counters','selected_app']]
    assert models['identities']['selected']==current and models['identities']['candidate']==candidate
    assert baseline['performance']['baseline_valid']
    assert current['Lokad.Onnx.dll']['sha256']=='49c3a958850d3e57daa2b7e29e6bd15ff9fc4f27af098d8ce44d8f20b9e065e8'
    assert current['Lokad.Onnx.Data.dll']['sha256']=='a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
    assert candidate['Lokad.Onnx.dll']['sha256']=='82c02785506b540d3fe590d48b0fdb15fc4b744ffb376cd626d758d5755bb16f'
    assert candidate['Lokad.Onnx.Data.dll']['sha256']=='3f80f8cb31f4a7e33fdb07247e3e1ed297ac53f7d8e3ebeb272980ff316f8236'
    assert selected['identities']['candidate']==current
    assert selected['performance']['admitted'] and read(base/'evidence/selected_app/closed.json')['admitted']
    assert len(selected['performance']['controls'])==63 and all(r['passed'] for r in selected['performance']['controls'])
    assert contracts['product']==census['product']==candidate
    build_path=base/'evidence/contracts/build-review.json';build=read(build_path)
    assert build['passed'] and not build['product_rebuilt'] and build['product']==candidate
    assert contracts['compiled_review']==census['original_compiled_review']==pin(build_path)
    assert [(r['original_methods'],r['unchanged'],len(r['differences'])) for r in build['methods']]==[(3277,3276,1),(697,697,0)]
    assert 'PrepareOwnedMatMulWeights' in build['methods'][0]['differences'][0]
    assert all(not r['added'] and not r['added_attributes'] for r in build['methods'])
    assert [(s['mode'],s['passed'],s['skipped']) for s in contracts['suites']]==[('512',27,0),('256',27,0),('scalar',2,0)]
    assert census['contracts']==pin(base/'evidence/contracts/closed.json')
    assert [r['mode'] for r in census['modes']]==['512','256']
    for item in census['modes']:
        r=item['result'];assert r['passed'] and r['owned_count']==87 and r['owned_bytes']==1459617792
        assert r['retained_maps']==37 and r['retained_clone_bytes']==268435456 and r['initializer_count']==649
        assert r['public_request_passed'] and r['idempotent'] and r['identities_preserved'] and r['logical_hashes_exact'] and r['pcm_unchanged'] and r['contexts_share_weights']
        assert not r['forced_gc'] and not r['product_rebuilt'] and not r['application_scored']
    assert counters['products']==models['identities'] and not counters['application_scored'] and not counters['release_admitted']
    assert counters['model_closure']==pin(base/'evidence/models/closed.json') and counters['census_closure']==pin(base/'evidence/census/closed.json')
    assert counters['build_review']==pin(base/'evidence/counters/build-review.json')
    counter_build=read(base/'evidence/counters/build-review.json')
    assert counter_build['passed'] and counter_build['binding_only'] and not counter_build['consumer_rebuilt'] and not counter_build['product_rebuilt']
    counter_spec=read(base/'evidence/counters/spec.json')
    initial=counters['initial'];assert initial==counter_spec['initial'] and not counters['completed_control_repeated']
    assert initial['original_code']==1 and initial['completed_job']=='selected-512' and initial['remaining_jobs']==['candidate-512','selected-256','candidate-256']
    assert [r['mode'] for r in counters['comparisons']]==['512','256']
    for comparison in counters['comparisons']:
        assert comparison['avoided_packs']==1740 and comparison['reconstructions']==609
        assert comparison['scratch_reduction']==29192355840 and comparison['copy_increase']==10217324544
        assert len(comparison['clips'])==20 and all(r['outputs_exact'] for r in comparison['clips'])
        for row in comparison['clips']:
            odd=row['frames']%2!=0 and row['frames']%3!=0
            assert row['avoided_packs']==87 and row['reconstructions']==(87 if odd else 0)
            assert row['scratch_reduction']==1459617792 and row['copy_increase']==(1459617792 if odd else 0)
    assert spec['failed_release_controls']==contracts['failed_release_controls']==census['failed_release_controls']==counters['failed_release_controls']
    assert not spec['release_admitted']
    assert models['consumers']['AudioBenchmark']==spec['consumers']['AudioBenchmark']==baseline['consumers']['AudioBenchmark']
    assert set(models['results'])=={f'{role}-{mode}-{isa}' for role in ['selected','candidate'] for mode in ['native','public'] for isa in ['512','256']}
    for role,original in [('current','selected'),('candidate','candidate')]:
        for isa in ['512','256']:
            native_result=models['results'][original+'-native-'+isa]
            native=native_result['native'];public=models['results'][original+'-public-'+isa]
            assert native_result['passed'] and native['audit_consistent'] and native['application_passed']
            assert native['numeric_gate_passed'] and not native['failures'] and (native['arrays'],native['values'])==(784,3090494)
            if role=='candidate':
                assert len(native['exact_selected_comparisons'])==784 and all(r['bit_identical'] for r in native['exact_selected_comparisons'])
                assert public['complete_selected_results_exact']
            assert public['passed'] and public['public_requests']==20
        public=models['results'][original+'-public-512']
        assert pin(base/'evidence'/(role+'-public.json'))==public['result']
        reference=read(base/'evidence'/(role+'-public.json'))
        assert len(reference['records'])==20 and reference['held_outputs_unchanged']
        assert reference['core_sha256']==spec['identities'][role]['Lokad.Onnx.dll']['sha256']
        assert reference['data_sha256']==spec['identities'][role]['Lokad.Onnx.Data.dll']['sha256']
        assert reference['flags']=={} and reference['runner_sha256']==spec['consumers']['AudioBenchmark']['sha256']
    return dict(passed=True,retained=spec['prerequisites'])


def validate(value, manifest, spec, native, original_protocol, reference, manifest_pin):
    original_protocol.validate_records(value, manifest, 'timing')
    assert manifest['family'] == 'parakeet' and len(manifest['cases']) == 20
    assert value['manifest_sha256'] == manifest_pin['sha256']
    assert value['engine'] == ('ort' if native else 'managed') and len(value['records']) == 80
    if native:
        assert value['python_binary'] == spec['interpreter']
        assert value['runner_sha256'] == spec['files']['runtime/native.py']['sha256']
        assert value['adapter_sha256'] == manifest['adapter']['sha256']
        assert value['versions'] == manifest['native_versions'] and value['native_binaries'] == manifest['native_binaries']
        assert value['native_settings'] == dict(provider='CPUExecutionProvider', intra_threads=1, inter_threads=1, sequential=True, graph_optimizations='all', spinning=False)
        assert value['numeric_libraries']
        for path, wanted in value['numeric_libraries'].items(): assert spec['external'][path] == wanted, path
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            assert value['flags'][key] == '1'
    else:
        assert value['runtime'] == '.NET 10.0.8' and value['processor_count'] == 1 and not value['flags']
        assert value['runner_sha256'] == spec['consumers']['AudioBenchmark']['sha256']
        assert all(value[k] == manifest[k] for k in ['core_sha256', 'data_sha256'])
        expected = {r['name']:r['result'] for r in reference['records']}
        assert len(expected) == 20 and set(expected) == {c['name'] for c in manifest['cases']}
        assert all(row['result'] == expected[row['name']] for row in value['records'])
    return dict(passed=True, requests=80, complete_retained_results_exact=not native)


def qualify(base, name, spec):
    folder = base/name/'output'; value = read(folder/'result.json')
    role=name.split('-')[-1];assert role in ['current','candidate','ort']
    reference_role='current' if role=='ort' else role
    manifest_path=base/'manifests'/(reference_role+'-parakeet.json')
    result = validate(value, read(manifest_path), spec, name.endswith('-ort'), records_protocol(base),
                      read(base/'evidence'/(reference_role+'-public.json')), pin(manifest_path))
    assert {p.name for p in folder.iterdir()} == {'result.json'} | {f'{i:03}.json' for i in range(80)}
    for index, row in enumerate(value['records']): assert row == read(folder/f'{index:03}.json')
    return dict(**result, result=pin(folder/'result.json'))


def evaluate(table):
    assert len(table)==21 and len({r['name'] for r in table})==21 and sum(r['is_corpus'] for r in table)==1
    corpus=next(r for r in table if r['is_corpus']);assert corpus['audio_seconds']==213.265
    def exact(value):
        fraction=Fraction(**value['exact_mean']);assert fraction>0;return fraction
    controls=[];gates=[];ratios=[]
    for row in table:
        for role in ['current','candidate','ort']:
            means=[exact(p) for p in row[role]['processes']]
            assert len(means)==2 and exact(row[role])==sum(means)/2
            ratio=max(means)/min(means);limit=Fraction(110 if row['is_corpus'] else 120,100)
            controls.append(dict(name=row['name'],role=role,process_ratio=float(ratio),limit=float(limit),passed=ratio<=limit))
        ratio=exact(row['candidate'])/exact(row['current'])
        if not row['is_corpus']:
            gates.append(dict(name=row['name'],candidate_over_current=float(ratio),limit=1.05,passed=ratio<=Fraction(105,100)))
        ratios.append(dict(name=row['name'],candidate_over_current=float(ratio)))
    for role in ['current','candidate','ort']:
        assert exact(corpus[role])==sum(exact(r[role]) for r in table if not r['is_corpus'])
    ratio=exact(corpus['candidate'])/exact(corpus['current'])
    gates.append(dict(name='corpus-at-least-three-percent-gain',candidate_over_current=float(ratio),limit=.97,passed=ratio<=Fraction(97,100)))
    assert len(controls)==63 and len(gates)==21
    stable=all(r['passed'] for r in controls)
    return dict(admitted=stable and all(r['passed'] for r in gates),controls_passed=stable,controls=controls,gates=gates,ratios=ratios,
        parity_target_met=stable and exact(corpus['candidate'])/exact(corpus['ort'])<=Fraction(105,100),
        policy='Six fresh processes; corpus max/min <=1.10 and each clip <=1.20 for all three engines. Candidate corpus >=3% gain and no clip >5% slower. Every clock retained; no unchanged retry.')
