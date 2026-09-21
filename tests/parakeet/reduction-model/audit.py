import hashlib
import importlib.util
import json
from common import *
import numpy as np


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    assert not (BASE / 'analysis.json').exists() and not (BASE / 'closed.json').exists()
    prepared = read(BASE / 'prepared.json')
    assert prepared['passed']
    verify(prepared['files'])
    resources = []
    identities = []
    for state_name in ('build-state.json', 'processes.json'):
        state = read(BASE / state_name)
        assert state['complete'] and state['code'] == 0
        identities.append(state['supervisor'])
        for run in state['runs']:
            assert run['complete'] and run['code'] in run['expected']
            identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
            samples = [json.loads(s) for s in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
            preflight, rss, seconds = (10, 8, 1200) if run['name'] == 'native' else (14, 12, 1200) if run['name'] == 'public' else (4, 2, 600)
            assert run['preflight']['available'] >= preflight * 1024**3
            assert len(samples) == run['samples'] > 0 and max(s['rss'] for s in samples) == run['peak_rss']
            assert all(s['rss'] < rss * 1024**3 and s['seconds'] < seconds and s['available'] >= 1024**3
                       and s['disk'] >= 20 * 1024**3 and s['output_bytes'] <= 1024**3
                       and all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in s['members']) for s in samples)
            if state_name == 'processes.json':
                assert all(len(s['members']) <= 1 for s in samples)
            resources.append(dict(name=run['name'], samples=len(samples), peak_rss=run['peak_rss'], exit_code=run['code']))
    for identity in identities:
        terminal(identity)
    assert [r['name'] for r in read(BASE / 'processes.json')['runs']] == ['native', 'public']
    for name, count in [('geometry', 416), ('hardware-off', 4)]:
        result = read(BASE / (name + '.json'))
        assert result['passed'] and result['tests'] == count and result['core_sha256'] == prepared['core']['sha256']
        assert result['runtime'] == '10.0.12' and result['affinity'] == 4
    geometry = read(BASE / 'geometry.json')
    assert geometry['eligible'] == geometry['changed'] == 64 and geometry['fallback'] == 352 and geometry['actual_operand_controls'] == 4
    instructions = read(BASE / 'instructions.json')
    assert instructions['passed']
    core, data = instructions['observations']
    assert len(core['changed_methods']) == 2 and len(core['added_methods']) == 1
    assert not data['changed_methods'] and not data['added_methods']

    native_auditor = module('original_native_auditor', ROOT / 'tests/parakeet/transcribe/audit.py')
    baseline = native_auditor.audit(REFERENCE, BASELINE)
    native = native_auditor.audit(REFERENCE, BASE / 'native/result.json')
    assert len(baseline['failures']) == 3 and baseline['maximum'] == 0.0002321004867553711
    assert native['arrays'] == 784 and native['values'] == 3090494
    native_result = read(BASE / 'native/result.json')
    assert native_result['core_sha256'] == prepared['core']['sha256'] and native_result['data_sha256'] == prepared['data']['sha256']
    native_run = read(BASE / 'processes.json')['runs'][0]
    assert native_run['code'] == (0 if native['numeric_gate_passed'] else 1)
    key = lambda r: (r['case'], r['label'], r['output'])
    old_keys, new_keys = {key(r) for r in baseline['failures']}, {key(r) for r in native['failures']}
    retained = read(BASELINE)
    changed_arrays = unchanged_arrays = 0
    for old_row, new_row in zip(retained['rows'], native_result['rows'], strict=True):
        assert old_row['name'] == new_row['name'] and old_row['actual'] == new_row['actual']
        for old, new in zip(old_row['comparisons'], new_row['comparisons'], strict=True):
            assert (old['label'], old['output'], old['shape'], old['dtype']) == (new['label'], new['output'], new['shape'], new['dtype'])
            old_bits = (Path(str(BASELINE) + '.tensors') / old['file']).read_bytes()
            new_bits = (BASE / 'native/result.json.tensors' / new['file']).read_bytes()
            assert hashlib.sha256(old_bits).hexdigest() == old['sha256']
            if old_bits == new_bits:
                unchanged_arrays += 1
            else:
                changed_arrays += 1
    assert changed_arrays + unchanged_arrays == 784
    public_auditor = module('original_public_auditor', ROOT / 'tests/audio/comparison/audit.py')
    manifest = read(CORPUS)
    for case in manifest['cases']:
        pcm = np.load(ROOT / case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    result = read(BASE / 'public/output/result.json')
    public_auditor.validate_worker(result, manifest, 'conformance')
    assert result['engine'] == 'managed' and result['runtime'] == '.NET 10.0.12' and result['flags'] == {} and result['processor_count'] == 1
    assert result['core_sha256'] == prepared['core']['sha256'] and result['data_sha256'] == prepared['data']['sha256']
    assert result['runner_sha256'] == prepared['public']['sha256'] and result['manifest_sha256'] == pin(CORPUS)['sha256']
    assert [p.name for p in sorted((BASE / 'public/output').glob('[0-9][0-9][0-9].json'))] == [f'{i:03}.json' for i in range(20)]
    for i, row in enumerate(result['records']):
        assert read(BASE / 'public/output' / f'{i:03}.json') == row
    analysis = dict(evidence_passed=True, native=native, baseline=baseline, native_numeric_passed=native['numeric_gate_passed'],
                    new_failed_arrays=sorted(new_keys-old_keys), removed_failed_arrays=sorted(old_keys-new_keys),
                    changed_arrays=changed_arrays, unchanged_arrays=unchanged_arrays, public_requests=20,
                    geometry=geometry, hardware_off=read(BASE / 'hardware-off.json'),
                    unchanged_core_methods=core['unchanged_methods'], unchanged_data_methods=data['unchanged_methods'],
                    resources=resources, terminal_identities=identities,
                    scope='Isolated arithmetic candidate, original complete native trajectories and public corpus; no speed/AMD/product qualification')
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for p in [*BASE.rglob('*'), *TOOLS.iterdir()]:
        if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(ROOT).parts):
            files[rel(p)] = pin(p)
    save(BASE / 'closed.json', dict(evidence_passed=True, native_numeric_passed=native['numeric_gate_passed'], files=files, terminal_identities=identities))
    print(json.dumps({k: v for k, v in analysis.items() if k not in ('terminal_identities', 'geometry', 'hardware_off')}))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
