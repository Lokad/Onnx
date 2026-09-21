import hashlib
import json
import shutil
import traceback
from common import *


def main():
    closure = read(ORIGINAL / 'closed.json')
    assert pin(ORIGINAL / 'closed.json')['sha256'] == 'e92b9fa6eee822aa02265c7a69b0327eac57482495e9570c73db89d98034c89a'
    assert closure['qualified']
    verify(closure['files'])
    for identity in closure['identities']:
        terminal(identity)
    assert pin(PRODUCT / 'Lokad.Onnx.dll')['sha256'] == 'd1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    (BASE / 'source').mkdir()
    for p in TOOLS.iterdir():
        if p.suffix in ('.cs', '.csproj'):
            shutil.copy2(p, BASE / 'source' / p.name)
    original = read(ORIGINAL / 'manifest.json')
    weights = {}
    for kind, key in [('weight', 'projection.weight'), ('bias', 'projection.bias')]:
        entry = original['weights'][key]
        value = np.load(ROOT / entry['file'], allow_pickle=False)
        assert value.dtype == np.float32 and list(value.shape) == entry['shape']
        assert hashlib.sha256(value.tobytes()).hexdigest() == entry['raw_sha256']
        path = BASE / (kind + '.bin')
        with path.open('xb') as f:
            f.write(value.tobytes())
        weights[kind] = rel(path)
    routes = []
    for name in ROUTES:
        result = read(ORIGINAL / 'outputs' / name / 'result.json')
        entry = next(v for v in result['outputs'] if v['name'] == '/pre_encode/Reshape_output_0')
        path = ORIGINAL / 'outputs' / name / entry['file']
        value = array(path, entry)
        assert value.dtype == np.float32 and value.shape == (1, 74, 4096)
        routes.append(dict(id=name, input=rel(path)))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    try:
        project = BASE / 'source/Probe.csproj'
        feed = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
        flags = monitor.FLAGS + [f'-p:FrozenProductDirectory={PRODUCT}']
        for i, command in enumerate([
            ['dotnet', 'restore', project, '--source', feed, *flags],
            ['dotnet', 'build', project, '-c', 'Release', '--no-restore', *flags]
        ]):
            monitor.worker(state, BASE / 'build-state.json', 'build-' + str(i), command, ROOT, [0], 4, 2, 600, True, BASE / 'source')
        files = dict(closure['files'])
        for p in [ORIGINAL / 'closed.json', MONITOR, *TOOLS.iterdir(), *BASE.rglob('*'),
                  *(ROOT / 'artifacts/parakeet-reduction-source-20260921').iterdir(),
                  ROOT / 'artifacts/parakeet-matmul-source-20260921/sgemm.cpp']:
            if p.is_file() and p.name != 'build-state.json':
                files[rel(p)] = pin(p)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'build-state.json', state)
    files[rel(BASE / 'build-state.json')] = pin(BASE / 'build-state.json')
    save(BASE / 'prepared.json', dict(passed=True, root=str(ROOT), files=files, routes=routes, blocks=BLOCKS, **weights,
        limits=dict(preflight_gib=4, rss_gib=2, seconds=600),
        admission=dict(minimum_rms_reduction=2, maximum_scaled_ratio=1, maximum_failure_ratio=1, preferred_block=256),
        scope='Actual-input projection arithmetic; no timing, full model or product promotion'))
    print(json.dumps(dict(passed=True, prepared=pin(BASE / 'prepared.json'))))


if __name__ == '__main__':
    main()
