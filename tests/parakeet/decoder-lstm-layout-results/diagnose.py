"""Explain the frozen scalar failure using a separately executed baseline control."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/decoder-lstm-layout'))
from protocol import pin, read, save, check_sample
from checks import census, compiled, contracts

V3 = ROOT/'artifacts/parakeet-decoder-lstm-layout-contracts-v3-amd-20260927'
BASE = ROOT/'artifacts/parakeet-decoder-lstm-layout-scalar-baseline-20260927'


def failures(path):
    ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    return {r.attrib['testName']: r.find('.//t:Message', ns).text.strip()
        for r in ET.parse(path).getroot().findall('.//t:UnitTestResult', ns)
        if r.attrib['outcome'] != 'Passed'}


def retained(folder):
    collected = folder/'collected'; receipt = read(collected/'collection.json')
    transfer = read(folder/'collection-transfer.json')
    assert receipt['terminal'] and receipt['input_error'] is None
    assert transfer['passed'] and transfer['archive'] == pin(folder/'results.tar.gz')
    assert transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['payload'] == pin(folder/'payload.json') == pin(collected/'payload.json')
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    state = read(collected/'identity.json')
    assert state['complete'] and state['code'] == receipt['code']
    assert state['supervisor'] == read(folder/'deployment.json')
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b)
        for r in state['runs'] for p, b in r['members'].items()]
    resources = []
    for row in state['runs']:
        assert row['complete'] and row['preflight']['available'] >= 4*1024**3
        assert row['preflight']['tmpfs'] >= 1024**3 and row['seconds'] < 900
        samples = [json.loads(line) for line in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] and samples
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:
                assert row['members'][str(member['pid'])] == member['birth']
                assert row['affinities'][str(member['pid'])] == member['expected_affinity']
        assert max(s['rss'] for s in samples) == row['peak_rss']
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    return state, resources


def main():
    assert not (BASE/'closed.json').exists()
    assert pin(V3/'failed.json')['sha256'] == '59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'
    for name, wanted in read(V3/'failed.json')['files'].items(): assert pin(V3/name) == wanted, name
    state, resources = retained(V3)
    baseline, baseline_resources = retained(BASE)
    assert state['code'] == 1 and [r['code'] for r in state['runs']] == [0]*6+[1]
    assert baseline['code'] == 0 and len(baseline['runs']) == 1
    stage = read(V3/'bundle/stage.json'); built = read(V3/'collected/built.json')
    scope = compiled(read(V3/'collected/inventory/instructions.json'), stage['current_product'], built['candidate'])
    assert scope == read(V3/'collected/inventory/review.json')
    original = set(stage['expected_census']); added = set(stage['added_tests'])
    assert len(original) == 172 and len(added) == 4
    values = []; modes = {}
    for mode in ['normal', 'noavx512', 'scalar']:
        folder = V3/'collected'/('contracts-'+mode)
        row = next(r for r in state['runs'] if r['name'] == 'contracts-'+mode)
        actual = census(folder/'contracts.trx', False)
        assert set(actual) == original | added and all(actual[n] == 'Passed' for n in added)
        loaded, projections = read(folder/'loaded.json'), read(folder/'projections.json')
        assert loaded['passed'] and loaded['mode'] == mode and str(loaded['pid']) in row['members']
        assert loaded['core_sha256'] == built['candidate']['Lokad.Onnx.dll']['sha256']
        assert loaded['consumer_sha256'] == built['consumer']['sha256']
        assert loaded['avx512'] == (mode == 'normal') and loaded['hardware'] == (mode != 'scalar')
        assert loaded['runtime'] == '10.0.8' and loaded['affinity'] == 4
        assert loaded['block'] == 4*loaded['vector_count']
        assert projections['passed'] and projections['calls'] == 380 and projections['projections'] == 760
        assert projections['values'] == 1945600 and len(projections['hashes']) == 760
        if mode != 'scalar':
            assert contracts(folder, stage, built, row) == read(folder/'review.json')
        values.append(projections['hashes'])
        modes[mode] = dict(passed=sum(v == 'Passed' for v in actual.values()),
            failed=sum(v != 'Passed' for v in actual.values()), added_passed=4,
            projection_values=projections['values'], projections=760, loaded=loaded,
            trx=pin(folder/'contracts.trx'))
    assert values[0] == values[1] == values[2]
    candidate_failures = failures(V3/'collected/contracts-scalar/contracts.trx')
    baseline_failures = failures(BASE/'collected/baseline-scalar/contracts.trx')
    assert candidate_failures == baseline_failures and len(candidate_failures) == 18
    baseline_cases = census(BASE/'collected/baseline-scalar/contracts.trx', False)
    candidate_cases = census(V3/'collected/contracts-scalar/contracts.trx', False)
    assert baseline_cases == {n: candidate_cases[n] for n in original}
    result = read(BASE/'collected/baseline-scalar/result.json')
    assert result['passed'] and result['diagnostic_only'] and result['test_exit_code'] == 1
    assert (result['passed_cases'], result['unsupported_cases']) == (154, 18)
    assert result['expected_rejections'] == baseline_failures
    row = baseline['runs'][0]
    assert row['code'] == 0 and row['exitcodes'] == {'worker': 0}
    assert row['processes']['worker']['pid'] == result['pid']
    for pid, entry in result['loaded'].items(): assert row['members'][pid] == entry['birth']
    for name, wanted in read(BASE/'bundle/prospective.json')['runtime'].items():
        assert pin(BASE/'collected/runtime'/name) == wanted, name
    unchanged = ['src/Lokad.Onnx/TensorExecutionOptions.cs', 'tests/Lokad.Onnx.Backend.Tests/LstmReferenceTests.cs']
    for name in unchanged: assert pin(V3/'collected/source'/name) == pin(ROOT/name)
    inventory = read(V3/'collected/inventory/instructions.json')['observations'][0]
    validate = 'Lokad.Onnx.TensorExecutionOptions::Validate::Void Validate()'
    assert validate in inventory['normalized_methods']
    assert validate not in inventory['differences'] and validate not in inventory['removed']
    assert not inventory['removed'] and inventory['unchanged_methods'] == 3282
    analysis = dict(contract_regression_found=False, original_campaign_passed=False,
        original_failure=pin(V3/'failed.json'), performance_measured=False,
        baseline=stage['current_product'], candidate=built['candidate'], compiled=scope,
        modes=modes, matched_existing_scalar_cases=172, matched_scalar_rejections=18,
        scalar_baseline_result=pin(BASE/'collected/baseline-scalar/result.json'),
        total_projection_values=3*1945600, projection_hashes_equal_across_modes=True,
        diagnosis='Both products reject the same 18 explicit-FMA requests when hardware intrinsics are disabled, before prepared dispatch. All four new facts pass in all modes. The original campaign remains failed; this separate baseline control resolves its cause.',
        resources=resources, baseline_resources=baseline_resources,
        candidate_model_and_performance_qualification_pending=True)
    save(BASE/'analysis.json', analysis)
    save(BASE/'closed.json', dict(passed=True, diagnostic_only=True, original_campaign_passed=False,
        analysis=pin(BASE/'analysis.json'), files={p.relative_to(BASE).as_posix(): pin(p)
            for p in BASE.rglob('*') if p.is_file()}))
    report = Path(__file__).with_name('contracts-diagnosis-20260927.json')
    assert not report.exists()
    save(report, dict(**analysis, baseline_diagnostic_closure=pin(BASE/'closed.json')))
    print(json.dumps(dict(diagnostic_passed=True, closure=pin(BASE/'closed.json'), report=pin(report))))


if __name__ == '__main__': main()
