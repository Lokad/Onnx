"""Audit each complete original/recovery role, then retain the joint verdict."""
import csv
import importlib.util
import json
import math
import sys

from run import ROOT, TOOLS, OLD_TOOLS, OLD, original, paths, prepared, initial, pin, read, write
from analyze import role, controls, families
from reference import references


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def check(mode):
    base = OLD if mode == 'clock' else paths(mode)[0]
    if mode == 'clock':
        initial()
    else:
        prepared(mode)
    folder = base / 'capture-collected'
    spec = read(base / 'bundle/spec.json')
    receipt, transfer = read(folder / 'capture-collection.json'), read(base / 'capture-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(base / 'capture-results.tar.gz')
    assert transfer['collection'] == pin(folder / 'capture-collection.json')
    assert receipt['terminal'] and receipt['code'] == (1 if mode == 'clock' else 0)
    for name, wanted in receipt['files'].items():
        assert pin(folder / name) == wanted, name
    assert pin(folder / 'spec.json') == pin(base / 'bundle/spec.json')
    for name, wanted in spec['files'].items():
        assert pin(folder / name) == wanted, name
    state = read(folder / 'capture-state.json')
    assert state['complete'] and state['code'] == receipt['code'] and receipt['state'] == pin(folder / 'capture-state.json')
    assert state['supervisor'] == read(base / 'capture-deployment.json')
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p,b in r['members'].items()]
    assert [r['name'] for r in state['runs']] == (['clock', 'stages'] if mode == 'clock' else [mode])
    run = state['runs'][0]
    assert run['name'] == mode and run['complete'] and run['code'] == 0
    original_folder = OLD / 'capture-collected'
    review = read(OLD / 'build-review.json'); built = read(original_folder / 'built.json')
    assert review['passed'] and review['arithmetic_equivalent'] and review['source_changes_exact']
    assert review['built'] == pin(original_folder / 'built.json')
    assert pin(original_folder / 'build-review.json') == pin(OLD / 'build-review.json')
    assert built['consumer'] == spec['consumer']
    for name, wanted in built['runtime_files'].items():
        assert pin(original_folder / name) == wanted, name
    reference = references()
    assert spec['diagnostic_references'] == reference['inputs']
    assert read(original_folder / 'evidence/cost-reference.json') == reference
    isolated = original.qualify()
    assert spec['isolated_evidence'] == isolated['evidence'] and not spec['release_admitted'] and spec['diagnostic_only']
    protocol = load('recovery_public_protocol', original.APP / 'collected/runtime/protocol.py')
    accounting = load('recovery_cpu_accounting', original.APP / 'collected/runtime/campaign_processes.py')
    manifest_path = original.APP / 'collected' / original.MANIFEST
    manifest = read(manifest_path); limits = spec['capture_limits']
    assert run['seconds'] < limits['seconds']
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
    gaps = [samples[0]['seconds']] + [b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])] + [run['seconds']-samples[-1]['seconds']]
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
    baseline = read(original_folder / 'clock/result.json')
    for index, row in enumerate(result['records']):
        assert row == read(folder / mode / f'{index:03}.json')
        assert row['thread_id'] == ready['thread_id']
        assert row['result'] == baseline['records'][index]['result'] and row['input_sha256'] == baseline['records'][index]['input_sha256']
    observed = role(result, folder / mode, mode, reference['graphs'], reference['routes'], reference['groups'], reference['numeric_ops'])
    corpus = sum(r['seconds'] for r in result['records'] if r['phase'] == 'measured') / 3
    assert math.isclose(observed['corpus_seconds'], corpus, rel_tol=1e-13)
    return dict(passed=True, mode=mode, role=observed, release_admitted=False, diagnostic_only=True,
        state=pin(folder / 'capture-state.json'), collection=pin(folder / 'capture-collection.json'),
        transfer=pin(base / 'capture-transfer.json'), initial=initial(), build_review=pin(OLD / 'build-review.json'),
        resources=dict(mode=mode, samples=len(samples), peak_rss=max(s['rss'] for s in samples), seconds=run['seconds']),
        terminal_owners=receipt['identities'], auditor=pin(__file__))


def main(mode):
    if mode in ['stages', 'markers']:
        base = paths(mode)[0]
        assert not (base / 'role-review.json').exists()
        value = check(mode)
        write(base / 'role-review.json', value)
        print(json.dumps(dict(mode=mode, passed=True, corpus=value['role']['corpus_seconds'], resources=value['resources'], review=pin(base / 'role-review.json'))))
        return
    assert mode == 'joint'
    base = ROOT / 'artifacts/parakeet-feed-forward-cost-joint-20260925'
    assert not base.exists()
    completed = {'clock': check('clock')}
    pins = {}
    for name in ['stages', 'markers']:
        prepared(name)
        path = paths(name)[0] / 'role-review.json'
        value = read(path)
        assert value['passed'] and value['mode'] == name and value['auditor'] == pin(__file__)
        assert value['initial'] == completed['clock']['initial']
        assert value['transfer'] == pin(paths(name)[0] / 'capture-transfer.json')
        assert value['collection'] == pin(paths(name)[0] / 'capture-collected/capture-collection.json')
        completed[name] = value; pins[name] = pin(path)
    roles = {name:value['role'] for name,value in completed.items()}
    assert all(r['graph_accounting'] == roles['clock']['graph_accounting'] for r in roles.values())
    states = [read((OLD if name == 'clock' else paths(name)[0]) / 'capture-collected/capture-state.json') for name in completed]
    assert states[0]['ended'] < states[1]['started'] < states[1]['ended'] < states[2]['started']
    validity = controls(roles); groups = families(roles['markers']); isolated = original.qualify()
    analysis = dict(passed=True, diagnostic_only=True, no_application_score=True, new_candidate_selected=False,
        release_admitted=False, isolated_evidence=isolated['evidence'], failed_release_controls=isolated['failed_release_controls'],
        overhead_subtracted=False, references=references()['inputs'], native_profile_is_earlier=True,
        native_packed_buffers_not_dumped=True, per_node_copy_bytes_are_retained_evidence=True,
        current_graph_counters_equal=True, actual_kernel_leaf_sampled=False, requests=240,
        observed_feed_forward_calls=7680, roles=roles, families=groups, controls=validity,
        resources=[v['resources'] for v in completed.values()], initial=completed['clock']['initial'],
        original_clock_repeated=False, original_partial_stages_excluded=True,
        usable_for_candidate_selection=validity['usable_for_candidate_selection'])
    base.mkdir()
    write(base / 'clock-review.json', completed['clock'])
    write(base / 'analysis.json', analysis)
    with (base / 'feed-forward-calls.csv').open('x', encoding='utf8', newline='') as stream:
        rows = roles['markers']['projections']; writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    write(base / 'closed.json', dict(passed=True, usable_for_candidate_selection=validity['usable_for_candidate_selection'],
        analysis=pin(base / 'analysis.json'), clock_review=pin(base / 'clock-review.json'), role_reviews=pins,
        files={p.name:pin(p) for p in base.iterdir()}, auditor=pin(__file__), initial=completed['clock']['initial'],
        terminal_owners=[i for v in completed.values() for i in v['terminal_owners']]))
    print(json.dumps(dict(closed=pin(base / 'closed.json'), usable=validity['usable_for_candidate_selection'],
        corpus={k:v['corpus_seconds'] for k,v in roles.items()}, families=groups,
        repeatability_passed=sum(r['passed'] for r in validity['repeatability']), observer_effects=validity['observer_effects'])))


if __name__ == '__main__':
    main(sys.argv[1])
