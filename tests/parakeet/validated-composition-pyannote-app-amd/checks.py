"""Preserve closed qualification, native contracts and complete public requests."""
import importlib.util
from fractions import Fraction
from protocol import ROLES, TIMING_ROLES, pin, read
from meetings_audit import audit_meetings
from semantics import compare_records


def prereqs(base, spec):
    assert set(spec['prerequisites']) == {'product','models','parakeet','shared'}
    reports = {}
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name
        assert pin(folder/'closed.json') == wanted['closed']
        proof = read(folder/'closed.json'); assert proof['passed']
        assert pin(folder/'analysis.json') == wanted['analysis'] == proof['analysis']
        reports[name] = read(folder/'analysis.json'); assert reports[name]['passed']
    assert reports['product']['identities'] == spec['identities']
    for name in ['models','parakeet','shared']:
        assert reports[name]['identities'] == spec['identities']
    for role in ROLES:
        graph=reports['models']['results'][role]
        assert graph['passed'] and (graph['arrays'],graph['values'],graph['public_calls'])==(18,2917107,16)
        if role=='candidate':
            assert graph['complete_public_results_exact']
            assert all(r['bit_identical'] for r in graph['comparisons'] if r['reference']=='production')
        for isa in ['512','256']:
            native=reports['parakeet']['results'][role+'-native-'+isa]['native']
            assert native['numeric_gate_passed'] and (native['arrays'],native['values'])==(784,3090494)
            public=reports['parakeet']['results'][role+'-public-'+isa]
            assert public['passed'] and public['public_requests']==20
            if role=='candidate':
                assert public['complete_selected_results_exact']
                assert len(native['exact_selected_comparisons'])==784 and all(r['bit_identical'] for r in native['exact_selected_comparisons'])
        shared=[reports['shared']['results'][role+'-'+mode] for mode in ['shared','e5']]
        assert all(r['passed'] for r in shared)
        assert sum(r['arrays'] for r in shared)==166 and sum(r['values'] for r in shared)==5000814
        if role=='candidate':assert all(r['exact_selected'] for value in shared for r in value['rows'])
    return dict(passed=True, retained=spec['prerequisites'])


def records_protocol(base):
    module = importlib.util.spec_from_file_location('original_audio_records',base/'runtime/protocol.py')
    result = importlib.util.module_from_spec(module); module.loader.exec_module(result); return result


def qualify(base, name, spec):
    if name == 'meetings-inputs':
        value = read(base/name/'output/inputs.json'); manifest = read(base/'meetings/manifest.json')
        assert value['passed'] and value['affinity'] == 4
        assert value['cases'] == [dict(name=c['name'],samples=c['samples'],pcm_sha256=c['pcm_sha256']) for c in manifest['cases']]
        return dict(passed=True, inputs=pin(base/name/'output/inputs.json'))
    if name == 'meetings-run':
        value = audit_meetings(base,base); assert value['passed']; return value
    cross_product = None
    native = name.startswith('native-') or name.endswith('-ort')
    mode = 'conformance' if name.startswith('native-') else 'timing'
    family = name.removeprefix('native-') if mode == 'conformance' else 'pyannote'
    role = 'selected' if native else name.split('-')[-1]
    manifest_path = base/'manifests'/(role+'-'+family+'.json'); manifest = read(manifest_path)
    folder = base/name/'output'; value = read(folder/'result.json')
    records_protocol(base).validate_records(value,manifest,mode)
    assert value['manifest_sha256'] == pin(manifest_path)['sha256']
    assert value['engine'] == ('ort' if native else 'managed')
    assert {p.name for p in folder.iterdir()} == {'result.json'} | {f'{i:03}.json' for i in range(len(value['records']))}
    for i,row in enumerate(value['records']): assert row == read(folder/f'{i:03}.json')
    if native:
        assert value['python_binary'] == spec['interpreter']
        assert value['runner_sha256'] == spec['files']['runtime/native.py']['sha256']
        assert value['adapter_sha256'] == manifest['adapter']['sha256']
        assert value['versions'] == manifest['native_versions'] and value['native_binaries'] == manifest['native_binaries']
        assert value['native_settings'] == dict(provider='CPUExecutionProvider',intra_threads=1,inter_threads=1,sequential=True,graph_optimizations='all',spinning=False)
        assert value['numeric_libraries']
        for path,wanted in value['numeric_libraries'].items(): assert spec['external'][path] == wanted,path
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']: assert value['flags'][key] == '1'
    else:
        assert value['runtime'] == '.NET 10.0.8' and value['processor_count'] == 1 and not value['flags']
        assert value['runner_sha256'] == spec['consumers']['AudioBenchmark']['sha256']
        assert all(value[k] == manifest[k] for k in ['core_sha256','data_sha256'])
        if name != 'timing-00-selected':
            selected = read(base/'timing-00-selected/output/result.json')
            cross_product = compare_records(value['records'], selected['records'], exact=True)
        if name == 'timing-04-candidate':
            first = read(base/'timing-01-candidate/output/result.json')
            compare_records(value['records'], first['records'], exact=True)
    expected = (4 if family == 'pyannote' else 20) if mode == 'conformance' else 16
    assert len(value['records']) == expected
    return dict(passed=True,requests=expected,result=pin(folder/'result.json'),cross_product=cross_product,maximum_centroid_error=max(r['maximum_centroid_error'] for r in value['records']))


def timing_table(results, manifest):
    assert len(results) == len(TIMING_ROLES)
    table = []
    for case in manifest['cases']:
        means = {}; row = dict(name=case['name'],audio_seconds=case['samples']/16000)
        for role in [*ROLES,'ort']:
            processes = []; values = []
            for index,(assigned,result) in enumerate(zip(TIMING_ROLES,results,strict=True)):
                if role != assigned: continue
                ticks = [Fraction(r['end_ticks']-r['start_ticks'],r['frequency']) for r in result['records'] if r['name'] == case['name'] and r['phase'] == 'measured']
                assert len(ticks) == 3 and all(t > 0 for t in ticks)
                values.extend(ticks); processes.append(dict(index=index,seconds=[float(t) for t in ticks],mean=float(sum(ticks)/3),exact_mean=dict(numerator=(sum(ticks)/3).numerator,denominator=(sum(ticks)/3).denominator)))
            assert len(values) == 6; means[role] = sum(values)/6
            row[role] = dict(seconds=float(means[role]),exact_mean=dict(numerator=means[role].numerator,denominator=means[role].denominator),rtf=float(means[role]/Fraction(case['samples'],16000)),minimum=float(min(values)),maximum=float(max(values)),processes=processes)
        row['ratios_to_ort'] = {role:float(means[role]/means['ort']) for role in ROLES}
        row['ratios_to_selected'] = dict(candidate=float(means['candidate']/means['selected']))
        table.append(row)
    return table
