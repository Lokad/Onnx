"""Reuse exact qualified binaries and both complete conformance gates locally."""
import hashlib
import shutil
from common import *
from storage_contract import STORAGE, replace_pending


def conformance_gate():
    audit = original_audit()
    assert pin(OLD/'failure-closed-v2.json')['sha256'] == FAILURE
    closure = read(OLD/'failure-closed-v2.json')
    assert closure['closure_passed'] and not closure['campaign_passed']
    for name, expected in closure['files'].items():
        assert pin(ROOT/name) == expected, name
    collected = OLD/'collected'; frozen = read(collected/'frozen.json')
    manifest = read(collected/'manifests/whisper.json')
    run = read(collected/'campaign/identity.json')['runs'][0]
    assert (run['phase'],run['family'],run['engine']) == ('conformance','whisper','managed')
    folder = collected/run['output']; managed = read(folder/'worker/result.json')
    validate_records(managed, manifest, 'conformance')
    audit.worker_identity(managed, manifest, frozen, collected, 'managed')
    resources = audit.resource_records(run, [json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()])
    for i,row in enumerate(managed['records']):
        assert read(folder/'worker'/f'{i:03}.json') == row
    prior = read(OLD/'prior-native-gate.json')
    native_base = NATIVE/'collected'; native_frozen = read(native_base/'frozen.json')
    native_manifest = read(native_base/'manifests/whisper.json')
    assert pin(native_base/'frozen.json') == prior['frozen'] and pin(NATIVE/'failure-closed.json') == prior['closure']
    assert pin(native_base/'manifests/whisper.json') == prior['manifest']
    assert {k:v for k,v in native_manifest.items() if k not in ['product_source','core_sha256','data_sha256']} == {
        k:v for k,v in manifest.items() if k not in ['product_source','core_sha256','data_sha256']}
    native_run = prior['worker']; folder = native_base/native_run['output']; native = read(folder/'worker/result.json')
    assert pin(folder/'worker/result.json') == prior['result']
    validate_records(native, native_manifest, 'conformance')
    audit.worker_identity(native, native_manifest, native_frozen, native_base, 'ort')
    assert audit.resource_records(native_run, [json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]) == prior['resource']
    import numpy as np
    for i,(case,row) in enumerate(zip(native_manifest['cases'],native['records'],strict=True)):
        assert read(folder/'worker'/f'{i:03}.json') == row
        actual = np.load(folder/'worker'/(case['name']+'.features.npy'), allow_pickle=False)
        expected = np.load(native_base/'assets'/case['features']['path'], allow_pickle=False)
        assert actual.dtype == expected.dtype == np.float32 and actual.shape == expected.shape == (1,128,3000) and np.isfinite(actual).all()
        difference = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        assert row['frontend'] == dict(values=384000,max_abs=float(difference.max()),failed=int((difference>1e-5).sum()),
            bits_equal=actual.tobytes()==expected.tobytes(),sha256=hashlib.sha256(actual.tobytes()).hexdigest())
    return dict(passed=True, calls=40, managed=dict(run=run,resources=resources,result=pin(collected/run['output']/'worker/result.json')),
        native=prior, manifest=pin(collected/'manifests/whisper.json'), frozen=pin(collected/'frozen.json'), failure=pin(OLD/'failure-closed-v2.json'))


def main():
    assert not BASE.exists()
    tests = ROOT/'artifacts/audio-whisper-storage-tests-20260921.log'
    assert 'Ran 10 tests' in tests.read_text(encoding='utf-8-sig') and '\nOK' in tests.read_text(encoding='utf-8-sig')
    gate = conformance_gate(); previous = read(OLD/'prepared.json'); frozen = read(OLD/'frozen.json')
    assert previous['product_source'] == '0f86c5de5a8e55ad4677f2734e21aa58f71fc3c7'
    for name, expected in previous['qualified'].items():
        assert pin(ROOT/name) == expected, name
    for name, expected in previous['source_bridge'].items():
        assert pin(ROOT/name) == expected, name
    # Check document insertion now; preparation must not mutate the live scoreboard.
    replace_pending((ROOT/'BENCHMARK.md').read_text(encoding='utf8'), '### Audio: matched AMD Whisper baseline\n\nVerified future result.\n\n')
    files = {}; copies = {}
    for name, expected in frozen['files'].items():
        if name.startswith(('assets/', 'bin/')) or name in ['runtime/native.py','runtime/protocol.py','runtime/whisper_adapter.py','runtime/campaign_processes.py','manifests/whisper.json']:
            assert pin(OLD/'collected'/name) == expected, name
            copies[name] = expected
    BASE.mkdir(); payload = BASE/'payload'; payload.mkdir()
    for name in ['supervise.py','storage_contract.py']:
        target = payload/'runtime'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(TOOLS/name,target)
        files['runtime/'+name] = pin(target)
    write(payload/'conformance-gate.json', gate); files['conformance-gate.json'] = pin(payload/'conformance-gate.json')
    shutil.copyfile(ROOT/'.agent/m5-whisper-amd-storage-20260921.md', payload/'prospective-plan.md')
    files['prospective-plan.md'] = pin(payload/'prospective-plan.md')
    source_files = {p.relative_to(ROOT).as_posix():pin(p) for p in TOOLS.glob('*.py')}
    source_files.update({str(Path('tests/audio/amd-comparison')/name).replace('\\','/'):pin(ROOT/'tests/audio/amd-comparison'/name)
        for name in ['protocol.py','audit.py','deploy.py']})
    value = {k:v for k,v in frozen.items() if k not in ['files','source','prior_native']}
    value['external'] = dict(frozen['external'])
    for name, expected in read(OLD/'collected/collection.json')['files'].items():
        if name.startswith(gate['managed']['run']['output']+'/') or name in ['frozen.json','manifests/whisper.json']:
            value['external'][OLD_REMOTE+'/'+name] = expected
    value.update(protocol=PROTOCOL, source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=copies|files, storage=STORAGE, prior_conformance=pin(payload/'conformance-gate.json'))
    write(BASE/'frozen-template.json', value)
    write(BASE/'prepared.json', dict(passed=True, source=value['source'], product_source=value['product_source'], copies=copies,
        uploads=files, sources=source_files, prior_failure=gate['failure'], template=pin(BASE/'frozen-template.json'),
        bytes_to_stage=sum(p['bytes'] for p in value['files'].values()), conformance_calls=40,
        fresh_timing_calls=320, measured_calls=240, storage=STORAGE,
        validation={tests.relative_to(ROOT).as_posix():pin(tests)}))
    print(json.dumps(read(BASE/'prepared.json') | {'sources':'bound', 'copies':'bound', 'uploads':'bound'}, indent=2))


if __name__ == '__main__':
    main()
