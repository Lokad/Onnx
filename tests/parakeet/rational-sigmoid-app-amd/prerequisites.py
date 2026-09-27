"""Admit one application experiment without erasing the failed operator screen."""
import math
from protocol import pin, read


def verify_comparisons(rows):
    assert len(rows) == 784 and sum(r['values'] for r in rows) == 3090494
    assert len({(r['case'], r['label'], r['output']) for r in rows}) == 784
    for row in rows:
        assert row['dtype'] in ['Float','Int32','Int64']
        assert row['values'] == math.prod(row['shape'])
        assert math.isfinite(row['maximum_scaled_error']) and 0 <= row['maximum_scaled_error'] <= 1e-4
        if row['dtype'] != 'Float':
            assert row['bit_identical'] and row['maximum_scaled_error'] == 0


def eligibility(reports, spec):
    baseline, models, contracts, qualified, screen, memory = [reports[n] for n in
        ['baseline','models','contracts','qualified','screen','memory']]
    current, candidate = spec['identities']['current'], spec['identities']['candidate']
    assert current['Lokad.Onnx.dll']['sha256'] == '8bb22038d0b4c09b56b2cdae06c49c165b8e646bc73ca28ad400f4ace0bfc659'
    assert current['Lokad.Onnx.Data.dll']['sha256'] == 'd02dbf550d7a6ea0ddf24985ffff7b86db135035ce7fab31f4bab063e0090620'
    assert candidate['Lokad.Onnx.dll']['sha256'] == '946ddfb66492c48a0fc6078ecbe1957ac494ff5d9ff0be42259d70e66e3b1f24'
    assert candidate['Lokad.Onnx.Data.dll']['sha256'] == 'dbe959361209bbc20db9bd566f037f58e01b9cc40d807c5745dbe2f4e1c1aca6'
    assert models['identities'] == dict(selected=current,candidate=candidate)
    assert qualified['built'] == current and contracts['product'] == candidate
    assert screen['products'] == memory['products'] == {role:products['Lokad.Onnx.dll'] for role,products in spec['identities'].items()}
    assert baseline['performance']['baseline_valid']
    assert qualified['root_source_verified'] and qualified['package']['passed'] and qualified['consumer']['passed']
    assert qualified['inventory']['method_bodies_equal'] and qualified['inventory']['implementation_flags_equal']
    assert contracts['compiled_review'] == dict(bytes=4595,sha256='2619dee6c8f64ea91578d8c63665878f8ed1181da8cbf464dd0514b4af2092e7')
    assert [(s['mode'],s['passed'],s['skipped']) for s in contracts['suites']] == [('normal',12,0),('scalar',12,0)]
    for suite in contracts['suites']:
        sweep = suite['sweep']
        assert sweep['passed'] and sweep['checked_values'] == 2048769 and sweep['tolerance'] == 1e-6
        assert 0 <= sweep['maximum_absolute_error'] <= 1e-6
    assert not screen['admitted'] and len(screen['rows']) == 46
    assert len(screen['controls']) == 94
    failures = [r for r in screen['controls'] if not r['passed']]
    assert len(failures) == 13 and failures == spec['failed_component_controls']
    assert len([r for r in screen['rows'] if not r['passed']]) == 4
    assert [(g['name'],g['passed']) for g in screen['gates']] == [
        ('at-least-75-percent-corpus-weighted',False),
        ('strict-process-weighted-separation',True),
        ('all-cases-no-five-percent-regression',False)]
    assert memory['diagnostic_only'] and not memory['admitted'] and not memory['performance_admission_attempted']
    assert memory['no_clock_trimmed'] and memory['allocation_counters_outside_timer']
    assert not contracts['release_admitted'] and not spec['release_admitted']


def verify(base, spec):
    assert set(spec['prerequisites']) == {'baseline','models','contracts','qualified','screen','memory'}
    reports = {}
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name
        proof = read(folder/'closed.json')
        assert pin(folder/'closed.json') == wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json') == wanted['analysis'] == proof['files']['analysis.json']
        reports[name] = read(folder/'analysis.json')
        assert reports[name]['passed']
        if name in ['screen','memory']: assert not proof['admitted']
    eligibility(reports, spec)
    baseline, models = reports['baseline'], reports['models']
    assert models['consumers']['AudioBenchmark']==spec['consumers']['AudioBenchmark']==baseline['consumers']['AudioBenchmark']
    assert set(models['results'])=={f'{role}-{mode}-{isa}' for role in ['selected','candidate'] for mode in ['native','public'] for isa in ['512','256']}
    for role,original in [('current','selected'),('candidate','candidate')]:
        for isa in ['512','256']:
            native_result=models['results'][original+'-native-'+isa]
            native=native_result['native'];public=models['results'][original+'-public-'+isa]
            assert native_result['passed'] and native['audit_consistent'] and native['application_passed']
            assert native['numeric_gate_passed'] and not native['failures'] and (native['arrays'],native['values'])==(784,3090494)
            if role=='candidate':
                verify_comparisons(native['selected_comparisons'])
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
