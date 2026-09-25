"""Admit diagnostic accounting only with complete public, resource and cost checks."""
import csv
import importlib.util
import json
import math

from analyze import MODES, controls, families, role
from reference import references
from run import ROOT, BASE, APP, MANIFEST, pin, read, write, prerequisites


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    prerequisites()
    assert not (BASE / 'closed.json').exists() and not (BASE / 'analysis.json').exists(), 'Retain the first verdict'
    folder = BASE / 'capture-collected'
    spec = read(BASE / 'bundle/spec.json')
    receipt = read(folder / 'capture-collection.json')
    transfer = read(BASE / 'capture-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE / 'capture-results.tar.gz')
    assert transfer['collection'] == pin(folder / 'capture-collection.json')
    assert receipt['terminal'] and receipt['code'] == 0
    for name, wanted in receipt['files'].items():
        assert pin(folder / name) == wanted, name
    assert pin(folder / 'spec.json') == pin(BASE / 'bundle/spec.json')
    for name, wanted in spec['files'].items():
        assert pin(folder / name) == wanted, name
    state = read(folder / 'capture-state.json')
    assert state['complete'] and state['code'] == 0 and receipt['state'] == pin(folder / 'capture-state.json')
    assert state['supervisor'] == read(BASE / 'capture-deployment.json')
    assert [r['name'] for r in state['runs']] == MODES
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    review = read(BASE / 'build-review.json')
    assert review['passed'] and review['arithmetic_equivalent'] and review['source_changes_exact']
    assert review['built'] == pin(folder / 'built.json') and pin(folder / 'build-review.json') == pin(BASE / 'build-review.json')
    built = read(folder / 'built.json')
    assert built['consumer'] == spec['consumer']
    for name, wanted in built['runtime_files'].items():
        assert pin(folder / name) == wanted, name
    reference = references()
    assert spec['diagnostic_references'] == reference['inputs']
    assert read(folder / 'evidence/cost-reference.json') == reference
    protocol = load('cost_public_protocol', APP / 'collected/runtime/protocol.py')
    accounting = load('cost_cpu_accounting', APP / 'collected/runtime/campaign_processes.py')
    manifest_path = APP / 'collected' / MANIFEST
    manifest = read(manifest_path)
    resources, results, roles = [], {}, {}
    limits = spec['capture_limits']
    for run in state['runs']:
        mode = run['name']
        assert run['complete'] and run['code'] == 0 and run['seconds'] < limits['seconds']
        assert run['preflight']['available'] >= limits['available_before'] and run['preflight']['tmpfs'] >= limits['tmpfs_before']
        samples = [json.loads(s) for s in (folder / 'logs' / (mode + '.resources.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0
        for sample in samples:
            assert sample['seconds'] < limits['seconds'] and sample['rss'] < limits['rss']
            assert sample['available'] >= spec['minimum_free'] and sample['tmpfs'] >= spec['minimum_free']
            assert sample['output'] < spec['output_limit']
            assert sample['rss'] == sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                assert run['members'][str(member['pid'])] == member['birth']
                assert member['affinity'] == [2] and member['threads'] and all(t == [2] for t in member['threads'])
        gaps = [samples[0]['seconds']] + [b['seconds']-a['seconds'] for a, b in zip(samples, samples[1:])] + [run['seconds']-samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        expected_accounting = accounting.foreign_fraction(run['cpu_before'], run['cpu_after'], state['supervisor']['pid'])
        assert run['accounting'] == expected_accounting
        assert expected_accounting['valid'] and expected_accounting['foreign_cpu_fraction'] <= .01
        result = read(folder / mode / 'result.json')
        protocol.validate_records(result, manifest, 'timing')
        assert result['passed'] and not result['sampled'] and result['runtime'] == '.NET 10.0.8'
        assert result['processor_count'] == 1 and result['affinity'] == 4 and result['flags'] == {}
        assert result['core_sha256'] == (built['core'] if mode == 'markers' else spec['product']['Lokad.Onnx.dll'])['sha256']
        assert result['data_sha256'] == built['data']['sha256'] and result['runner_sha256'] == spec['consumer']['sha256']
        assert result['manifest_sha256'] == pin(manifest_path)['sha256']
        ready, released = read(folder / mode / 'ready.json'), read(folder / mode / 'release.json')
        assert ready == run['ready'] and ready['pid'] == released['pid'] == run['owner']['pid']
        assert not released['sampled'] and ready['affinity'] == 4 and not ready['flags'] and ready['warmup_records'] == 20
        assert abs(ready['birth_milliseconds']/1000-run['owner']['birth']) < 1.1
        assert len(result['records']) == 80 and result['held_outputs_unchanged']
        for index, row in enumerate(result['records']):
            assert row == read(folder / mode / f'{index:03}.json')
            assert row['thread_id'] == ready['thread_id']
            if mode != 'clock':
                baseline = results['clock']['records'][index]
                assert row['result'] == baseline['result'] and row['input_sha256'] == baseline['input_sha256']
        results[mode] = result
        roles[mode] = role(result, folder / mode, mode, reference['graphs'], reference['routes'],
                           reference['groups'], reference['numeric_ops'])
        corpus = sum(r['seconds'] for r in result['records'] if r['phase'] == 'measured') / 3
        assert math.isclose(roles[mode]['corpus_seconds'], corpus, rel_tol=1e-13)
        if mode != 'clock':
            assert roles[mode]['graph_accounting'] == roles['clock']['graph_accounting']
        resources.append(dict(mode=mode, samples=len(samples), peak_rss=max(s['rss'] for s in samples), seconds=run['seconds']))
    validity = controls(roles)
    groups = families(roles['markers'])
    analysis = dict(passed=True, diagnostic_only=True, no_application_score=True, new_candidate_selected=False,
        overhead_subtracted=False, references=reference['inputs'],
        native_profile_is_earlier=True, native_packed_buffers_not_dumped=True,
        per_node_copy_bytes_are_retained_evidence=True, current_graph_counters_equal=True,
        actual_kernel_leaf_sampled=False, requests=240, observed_feed_forward_calls=7680,
        roles=roles, families=groups, controls=validity, resources=resources,
        usable_for_candidate_selection=validity['usable_for_candidate_selection'])
    write(BASE / 'analysis.json', analysis)
    with (BASE / 'feed-forward-calls.csv').open('x', encoding='utf8', newline='') as stream:
        rows = roles['markers']['projections']
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    write(BASE / 'closed.json', dict(passed=True, usable_for_candidate_selection=validity['usable_for_candidate_selection'],
        analysis=pin(BASE / 'analysis.json'), files=files, diagnostic_only=True,
        terminal_owners=receipt['identities'], auditor=pin(__file__)))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), corpus={k: v['corpus_seconds'] for k, v in roles.items()},
        usable_for_candidate_selection=validity['usable_for_candidate_selection'], families=groups,
        repeatability_passed=sum(r['passed'] for r in validity['repeatability']), observer_effects=validity['observer_effects'])))


if __name__ == '__main__':
    main()
