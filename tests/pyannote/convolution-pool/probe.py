"""Run the original captured-input probe with the new qualified Core identity."""
import json
import shutil
import traceback
import common
from phase_audit import audit_preparation

common.BASE = common.ROOT / 'artifacts/pyannote-convolution-pool-v4-20260921'
common.monitor.BASE = common.BASE
from common import *


def main():
    assert not (BASE / 'probe-processes.json').exists()
    preparation = audit_preparation(BASE, common)
    save(BASE / 'preparation-audit.json', preparation)
    reference = PROBE / 'closed.json'
    assert pin(reference)['sha256'] == 'd06bffe6d4aba50e7b17373c2a155234c514419bd4859063938fb1fe436fb626'
    old = read(reference)
    assert old['passed']
    verify(old['files'])
    prepared = read(BASE / 'prepared.json')
    source = BASE / 'probe-source'
    source.mkdir()
    original_tools = ROOT / 'tests/pyannote/context-reuse-probe'
    shutil.copy2(original_tools / 'Probe.csproj', source / 'Probe.csproj')
    program = (original_tools / 'Program.cs').read_text(encoding='utf8')
    for old_sha, name in [('469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd', 'Lokad.Onnx.dll'),
                          ('e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb', 'Lokad.Onnx.Data.dll')]:
        assert program.count(old_sha) == 1
        program = program.replace(old_sha, pin(BASE / 'runtime' / name)['sha256'])
    (source / 'Program.cs').write_text(program, encoding='utf8')
    shutil.copy2(PROBE / 'manifest.json', BASE / 'manifest.json')
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    save(BASE / 'probe-processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false', '-p:FrozenProductDirectory=' + str(BASE / 'runtime')]
    try:
        monitor.worker(state, BASE / 'probe-processes.json', 'probe-restore',
            ['dotnet', 'restore', source / 'Probe.csproj', *flags, '--source', FEED, '--packages', BASE / 'packages'], source, [0], 10, 8, 900, True, None)
        monitor.worker(state, BASE / 'probe-processes.json', 'build',
            ['dotnet', 'build', source / 'Probe.csproj', '-c', 'Release', *flags, '--no-restore', '--disable-build-servers', '-o', BASE / 'bin'], source, [0], 10, 8, 900, True, BASE / 'bin')
        for path in (BASE / 'runtime').glob('*.dll'):
            target = BASE / 'bin' / path.name
            if not target.exists():
                shutil.copy2(path, target)
            assert pin(target) == pin(path)
        files = dict(prepared['files'])
        files.update(old['files'])
        files[rel(reference)] = pin(reference)
        for path in [BASE / 'prepared.json', BASE / 'preparation-audit.json', BASE / 'manifest.json',
                     ROOT / 'tests/pyannote/context-reuse-probe/audit.py', *TOOLS.iterdir(), *source.iterdir(), *(BASE / 'bin').iterdir()]:
            if path.is_file():
                files[rel(path)] = pin(path)
        save(BASE / 'probe-prepared.json', dict(passed=True, files=files, orders=['forward', 'reverse'],
            limits=dict(preflight_gib=10, rss_gib=8, seconds=900),
            admission='Both orders must reduce embedding allocation versus the retained predecessor and preserve every bit, input and held result.',
            scope='Captured graph allocation qualification; no full-application or ORT timing claim.'))
        for order in ['forward', 'reverse']:
            verify(files)
            monitor.worker(state, BASE / 'probe-processes.json', order,
                ['dotnet', BASE / 'bin/Probe.dll', ROOT, BASE / 'manifest.json', BASE / order, order], ROOT, [0], 10, 8, 900, False, BASE / order)
            result = read(BASE / order / 'result.json')
            assert result['passed'] and len(result['records']) == 54
            print(order, '54 captured graph calls passed', flush=True)
        verify(files)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'probe-processes.json', state)


if __name__ == '__main__':
    main()
