"""Linux adaptation of the existing bounded complete-request capture protocol."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
DOTNET = '/home/vermorel/.dotnet/dotnet'
JOBS = ['control', 'sampled-a', 'sampled-b']
LIMITS = dict(seconds=900, preflight_available=10*1024**3, preflight_tmpfs=3*1024**3,
    rss=8*1024**3, available=1024**3, tmpfs=1024**3, output=1024**3, artifacts=2*1024**3)


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))


def pin(p):
    p = Path(p)
    with p.open('rb') as stream: return dict(bytes=p.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def save(p, value):
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8'); temporary.replace(p)


def live(identity):
    try:
        p = psutil.Process(identity['pid'])
        return p.create_time() == identity['birth'] and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess: return False


def terminal(identity): assert not live(identity), identity


def clean_env():
    env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    env.pop('PYTHONOPTIMIZE', None)
    env.update(TMPDIR=str(BASE / 'tmp'), PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    return env


def verify():
    spec = read(BASE / 'payload.json'); assert spec['limits'] == LIMITS
    for name, wanted in spec['files'].items():
        p = (BASE / name).resolve(); assert p.is_relative_to(BASE) and p != BASE
        assert pin(p) == wanted, name
    for name, wanted in spec['external'].items(): assert pin(name) == wanted, name
    return spec


def idle():
    # Azure host agents remain running. Reject an actual benchmark/runtime owner.
    active = []
    own = psutil.Process(); ancestors = {p.pid for p in own.parents()} | {own.pid}
    for p in psutil.process_iter(['pid', 'name', 'create_time', 'cmdline']):
        if p.pid in ancestors: continue
        name, command = p.info['name'], ' '.join(p.info['cmdline'] or [])
        if name in ('dotnet', 'perf') or (name.startswith('python') and ('/dev/shm/lokad-' in command or '/Onnx/' in command)):
            active.append(p.info)
    assert not active, ('Existing benchmark owner', active)


def thread_affinities(p):
    result = []
    for thread in p.threads():
        try: result.append(dict(id=thread.id, affinity=sorted(os.sched_getaffinity(thread.id))))
        except ProcessLookupError: pass
    assert result and all(t['affinity'] == p.cpu_affinity() for t in result), result
    return result


def artifact_size():
    return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())


def load_pair():
    # Preparation pins the old source and records every platform-only replacement.
    source = (BASE / 'tools/pair.py.txt').read_text(encoding='utf8')
    namespace = dict(globals())
    exec(compile(source, 'original_pair_with_linux_adapter', 'exec'), namespace)
    return namespace['pair']


def main():
    assert sys.platform == 'linux' and not (BASE / 'identity.json').exists()
    os.sched_setaffinity(0, {0}); idle(); spec = verify()
    assert psutil.boot_time() == spec['boot_time']
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert shutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    (BASE / 'logs').mkdir(); (BASE / 'tmp').mkdir()
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()),
        started=time.time(), boot_time=psutil.boot_time(), runs=[])
    save(BASE / 'identity.json', state)
    import public_audit
    manifest = read(BASE / 'manifest.json')
    for case in manifest['cases']: case['raw_sha256'] = spec['raw_pcm'][case['name']]
    expected = {r['name']:r['result'] for r in read(BASE / 'prior-amd-result.json')['records']}
    pair = load_pair()
    try:
        for name in JOBS:
            verify()
            assert artifact_size() <= LIMITS['artifacts'] and time.time()-state['started'] < 3600
            sampled = name != 'control'; output = BASE / name
            pair(state, BASE / 'identity.json', name, [DOTNET, BASE / 'runtime/SampledAudio.dll', BASE,
                BASE / 'manifest.json', output, 'timing', 'sampled' if sampled else 'control'], output, sampled, 10)
            result = read(output / 'result.json'); public_audit.validate_worker(result, manifest, 'timing')
            assert result['passed'] and result['sampled'] == sampled and result['runtime'] == '.NET 10.0.8'
            assert result['core_sha256'] == spec['core'] and result['data_sha256'] == spec['data']
            assert result['runner_sha256'] == spec['consumer']['sha256'] and result['manifest_sha256'] == pin(BASE / 'manifest.json')['sha256']
            assert all(row['result'] == expected[row['name']] for row in result['records'])
            assert read(output / 'ready.json')['warmup_records'] == 4
            print(name, 'all 16 complete public checks passed', flush=True)
        verify(); assert artifact_size() <= LIMITS['artifacts']; state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); traceback.print_exc()
    finally:
        state.update(complete=True, ended=time.time()); save(BASE / 'identity.json', state)
    return state['code']


if __name__ == '__main__': raise SystemExit(main())
