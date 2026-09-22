"""Selected versus prepared-convolution Parakeet regression; no timing score."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-parakeet-20260922'
MODEL = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922'
CONTROL = ROOT/'artifacts/pyannote-single-panel-composition-20260922'
REFERENCE = ROOT/'artifacts/parakeet-transcription-20260919/frozen/reference/manifest.json'
CORPUS = ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/parakeet.json'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
NATIVE_AUDITOR = ROOT/'tests/parakeet/transcribe/audit.py'
PUBLIC_AUDITOR = ROOT/'tests/audio/comparison/audit.py'
IDENTITIES = {
    'selected': ('1279b4b662241db2404fa4875eae20eaa924f15677a85655f99b8f81cd24b309', '4e602d9f6a35a51277d6deb9d75779d84cecf0a3a433d1b4eb70b0006462cca4'),
    'candidate': ('3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f', '6318cf48691470b908eec4c4d09c558172e43ce3b04bca9039c68966998a684b')}
KNOWN_FAILURES = ['english-16k/step-26/outputs', 'english-frame-limit/step-26/outputs', 'english-repeat/step-26/outputs']
JOBS = [(role+'-'+mode, role, mode, 12 if mode == 'native' else 14, 8 if mode == 'native' else 12) for role in IDENTITIES for mode in ['native', 'public']]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


monitor = module('parakeet_blocked_monitor', MONITOR); monitor.BASE = BASE
pin, read, save, verify, terminal = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal


def prepare():
    assert not BASE.exists()
    for folder, relative in [(CONTROL, True), (MODEL, False)]:
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin((ROOT if relative else folder)/name) == wanted, name
        for identity in proof['identities']: terminal(identity)
    assert pin(MODEL/'closed.json')['sha256'] == 'e7a9a30d88ef2a425c0c51b007e2b7d89428d54e50ef7dc4e446e191f95719e3'
    assert pin(REFERENCE)['sha256'] == '3bad7d262b8809b1265c84c8e66d02ee38e7d4cff2d92014448976a9e161103c'
    BASE.mkdir(); (BASE/'logs').mkdir()
    files = {p.relative_to(ROOT).as_posix(): pin(p) for p in [REFERENCE, CORPUS, MODEL/'closed.json', CONTROL/'closed.json', MONITOR, NATIVE_AUDITOR, PUBLIC_AUDITOR, *TOOLS.iterdir()] if p.is_file()}
    for role, folder in [('selected', CONTROL), ('candidate', MODEL)]:
        target = BASE/'runtime'/role; shutil.copytree(folder/'runtime', target)
        for consumer, old, sha in [
            ('TranscribeReplay', ROOT/'artifacts/parakeet-transcription-20260919/frozen/replay', '335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'),
            ('AudioBenchmark', ROOT/'artifacts/audio-ort-baseline-v2-20260919/bin', '7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1')]:
            assert pin(old/(consumer+'.dll'))['sha256'] == sha
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']: shutil.copy2(old/(consumer+'.'+suffix), target/(consumer+'.'+suffix))
        for filename, sha in zip(['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'], IDENTITIES[role], strict=True): assert pin(target/filename)['sha256'] == sha
        files.update({p.relative_to(ROOT).as_posix(): pin(p) for p in target.iterdir() if p.is_file()})
    ref = read(REFERENCE)
    for name, entry in ref['files'].items():
        p = REFERENCE.parent/name; assert pin(p) == {k:entry[k] for k in ['bytes', 'sha256']}; files[p.relative_to(ROOT).as_posix()] = pin(p)
    corpus = read(CORPUS)
    for entry in [*corpus['models'].values(), corpus['reference'], corpus['upstream'], *[c['pcm'] for c in corpus['cases']]]:
        p = ROOT/entry['path']; assert pin(p) == {k:entry[k] for k in ['bytes', 'sha256']}; files[p.relative_to(ROOT).as_posix()] = pin(p)
    ast.parse(Path(__file__).read_text())
    save(BASE/'prepared.json', dict(passed=True, files=files, identities=IDENTITIES, jobs=JOBS, known_native_failures=KNOWN_FAILURES,
        criterion='All outputs bit-identical to selected; original public contracts; exactly the three preserved Windows native failures with unchanged maxima; no application speed claim.'))
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'), files=len(files))), flush=True)


def execute():
    spec = read(BASE/'prepared.json'); assert spec['passed']; verify(spec['files'])
    assert not (BASE/'processes.json').exists()
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    try:
        for name, role, mode, minimum, rss in JOBS:
            out = BASE/name; out.mkdir(); runtime = BASE/'runtime'/role
            if mode == 'native': args = ['dotnet', runtime/'TranscribeReplay.dll', ROOT/'models/parakeet-tdt-0.6b-v3', REFERENCE, out/'result.json']
            else: args = ['dotnet', runtime/'AudioBenchmark.dll', ROOT, CORPUS, out/'output', 'conformance']
            row = monitor.worker(state, BASE/'processes.json', name, args, ROOT, [0, 1] if mode == 'native' else [0], minimum, rss, 1800, False, out)
            result = read(out/('result.json' if mode == 'native' else 'output/result.json'))
            assert (result['core_sha256'], result['data_sha256']) == IDENTITIES[role]
            if mode == 'native':
                assert result['application_passed'] and not result['errors'] and result['comparisons'] == 784 and result['values_compared'] == 3090494
                assert row['code'] == (0 if result['passed'] else 1)
            else: assert len(result['records']) == 20 and result['held_outputs_unchanged']
            print(name, 'complete', flush=True)
        verify(spec['files']); state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE/'processes.json', state)


def resource_audit():
    state = read(BASE/'processes.json'); assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == [j[0] for j in JOBS]
    ids, resources = [state['supervisor']], []
    for run, (name, role, mode, minimum, rss) in zip(state['runs'], JOBS, strict=True):
        assert run['complete'] and run['code'] in ([0, 1] if mode == 'native' else [0])
        assert run['preflight'] == run['preflight_observations'][-1] and run['preflight']['available'] >= minimum*1024**3
        ids.extend(dict(pid=int(pid), birth=birth) for pid,birth in run['members'].items())
        rows = [json.loads(s) for s in (BASE/'logs'/(name+'.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == run['samples'] > 0 and max(r['rss'] for r in rows) == run['peak_rss']
        assert rows[-1]['seconds'] <= run['seconds'] < 1800
        gaps = [rows[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(rows, rows[1:])]+[run['seconds']-rows[-1]['seconds']]
        assert all(0 <= x < 10 for x in gaps)
        for r in rows:
            assert r['rss'] < rss*1024**3 and r['available'] >= 1024**3 and r['disk'] >= 20*1024**3 and r['output_bytes'] <= 1024**3
            assert len(r['members']) <= 1 and r['rss'] == sum(p['rss'] for p in r['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in r['members'])
        resources.append(dict(name=name, samples=len(rows), seconds=run['seconds'], peak_rss=run['peak_rss'], code=run['code']))
    for identity in ids: terminal(identity)
    return ids, resources


def audit():
    import numpy as np
    assert not (BASE/'closed.json').exists()
    spec = read(BASE/'prepared.json'); verify(spec['files']); ids, resources = resource_audit()
    native_auditor = module('blocked_parakeet_native', NATIVE_AUDITOR)
    public_auditor = module('blocked_parakeet_public', PUBLIC_AUDITOR)
    corpus = read(CORPUS)
    for case in corpus['cases']:
        pcm = np.load(ROOT/case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    reports, native, public = {}, {}, {}
    for role in IDENTITIES:
        path = BASE/(role+'-native/result.json'); native[role] = read(path)
        report = native_auditor.audit(REFERENCE, path); reports[role] = report
        assert report['audit_consistent'] and report['application_passed'] and report['arrays'] == 784 and report['values'] == 3090494
        failures = {'/'.join(r[k] for k in ['case', 'label', 'output']):r for r in report['failures']}
        assert set(failures) == set(KNOWN_FAILURES) and not report['numeric_gate_passed']
        row = next(r for r in resources if r['name'] == role+'-native'); assert row['code'] == 1
        result = native[role]; assert (result['core_sha256'], result['data_sha256']) == IDENTITIES[role]
        assert result['runtime'] == '.NET 10.0.12' and not result['settings']
        assert result['runner_sha256'] == pin(BASE/'runtime'/role/'TranscribeReplay.dll')['sha256']
        save(BASE/(role+'-native-audit.json'), report)
        result = read(BASE/(role+'-public/output/result.json')); public[role] = result
        public_auditor.validate_worker(result, corpus, 'conformance')
        assert (result['core_sha256'], result['data_sha256']) == IDENTITIES[role]
        assert result['engine'] == 'managed' and result['runtime'] == '.NET 10.0.12' and not result['flags'] and result['processor_count'] == 1
        assert result['runner_sha256'] == pin(BASE/'runtime'/role/'AudioBenchmark.dll')['sha256'] and result['manifest_sha256'] == pin(CORPUS)['sha256']
        assert [p.name for p in sorted((BASE/(role+'-public/output')).glob('[0-9][0-9][0-9].json'))] == [f'{i:03}.json' for i in range(20)]
        for i,r in enumerate(result['records']): assert read(BASE/(role+f'-public/output/{i:03}.json')) == r
    comparisons = []
    for a,b in zip(native['selected']['rows'], native['candidate']['rows'], strict=True):
        assert a['name'] == b['name'] and a['actual'] == b['actual']
        for x,y in zip(a['comparisons'], b['comparisons'], strict=True):
            assert all(x[k] == y[k] for k in ['label', 'output', 'shape', 'dtype', 'file'])
            left = (BASE/'selected-native/result.json.tensors'/x['file']).read_bytes()
            right = (BASE/'candidate-native/result.json.tensors'/y['file']).read_bytes()
            assert left == right
            comparisons.append(dict(case=a['name'], label=x['label'], output=x['output'], values=len(left)//(8 if x['dtype'] == 'Int64' else 4), sha256=hashlib.sha256(left).hexdigest(), bit_identical=True))
    assert len(comparisons) == 784 and sum(c['values'] for c in comparisons) == 3090494
    assert reports['selected']['failures'] == reports['candidate']['failures']
    for a,b in zip(public['selected']['records'], public['candidate']['records'], strict=True): assert a['result'] == b['result']
    analysis = dict(passed=True, arrays=784, values=3090494, exact_public_requests=20, comparisons=comparisons, native=reports,
        known_native_failures_preserved=True, resources=resources, identities=ids, no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = dict(spec['files']); files.update({p.relative_to(ROOT).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE/'closed.json', dict(passed=True, files=files, identities=ids, candidate_native_numeric_passed=False))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), arrays=784, values=3090494, exact_public_requests=20, known_native_failures=KNOWN_FAILURES, resources=resources)), flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'execute', 'audit']
    globals()[sys.argv[1]]()
