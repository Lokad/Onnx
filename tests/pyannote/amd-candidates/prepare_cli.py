"""Retain the model payload and change only CLI test output draining in a successor."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import traceback
from candidate_protocol import pin, read, write, verified_files

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT/'artifacts/pyannote-amd-candidates-v2-20260921'
BASE = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921'
PAYLOAD = PRIOR/'payload'


def save(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8'); temporary.replace(path)


def main():
    sys.path.insert(0, str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages')); import psutil
    prepared = read(PRIOR/'prepared.json'); verified = read(PRIOR/'preparation-verified.json')
    assert verified['passed'] and verified['prepared'] == pin(PRIOR/'prepared.json')
    assert pin(PAYLOAD/'payload.json') == prepared['payload']
    assert pin(PRIOR/'payload.tar.gz') == prepared['archive']
    old = read(PAYLOAD/'payload.json'); verified_files(PAYLOAD, old['files'])
    assert psutil.virtual_memory().available >= 8*1024**3
    BASE.mkdir(); (BASE/'logs').mkdir(); payload = BASE/'payload'; payload.mkdir()
    for name in old['files']:
        target = payload/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(PAYLOAD/name, target)
    source = payload/'source'; relative = 'tests/Lokad.Onnx.Backend.Tests/CliExitCodeTests.cs'; target = source/relative
    before = target.read_text(); after = (ROOT/relative).read_text()
    expected = before.replace('        if (!p.WaitForExit(180000))',
        '        // Drain both pipes while the CLI runs; verbose output can exceed their\n'
        '        // capacity before the process reaches its exit-code assertion.\n'
        '        var stdout = p.StandardOutput.ReadToEndAsync();\n'
        '        var stderr = p.StandardError.ReadToEndAsync();\n'
        '        if (!p.WaitForExit(180000))').replace('        return p.ExitCode;',
        '        Task.WhenAll(stdout, stderr).GetAwaiter().GetResult();\n        return p.ExitCode;')
    assert after == expected and after != before; shutil.copy2(ROOT/relative, target)
    # Keep the model/runtime/consumer bytes fixed. Only one test source differs.
    spec = dict(old); spec['files'] = dict(old['files']); spec['source_files'] = dict(old['source_files'])
    spec['files']['source/'+relative] = pin(target); spec['source_files'][relative] = pin(target)
    spec['source'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    spec['test_harness_adaptation'] = dict(path=relative, before=old['source_files'][relative], after=pin(target),
                                          predecessor=pin(PRIOR/'prepared.json'))
    verified_files(payload, spec['files'])
    tool = ROOT/'artifacts/pyannote-lstm-panel-admission-v2-20260921/failed-tools/prepare.py'
    assert pin(tool)['sha256'] == read(ROOT/'artifacts/pyannote-lstm-panel-admission-v2-20260921/failed-preparation.json')['files']['failed-tools/prepare.py']['sha256']
    original = tool.read_text(); start = original.index('    own = psutil.Process();'); end = original.index('    try:\n        backend = Path(', start)
    monitor = textwrap.dedent(original[start:end]); scope = dict(globals(), psutil=psutil)
    exec(compile(monitor, str(tool)+' [preserved monitor]', 'exec'), scope)
    state = scope['state']; started = scope['started']
    try:
        projects = [('backend', source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
                    ('tensors', source/'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'),
                    ('cli', source/'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
                    ('il-bridge', payload/'il-bridge/IlBridge.csproj')]
        for label, project in projects: scope['restore_build'](label, project, source)
        scope['command']('il-bridge', ['dotnet', payload/'il-bridge/bin/Release/net10.0/IlBridge.dll',
            payload/'runtimes/rows', source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0', BASE/'logs/local-il-bridge.json'], source)
        bridge = read(BASE/'logs/local-il-bridge.json'); assert bridge['passed']
        assert [(r['assembly'], r['methods'], r['equal']) for r in bridge['observations']] == [
            ('Lokad.Onnx.dll', 3104, True), ('Lokad.Onnx.Data.dll', 691, True)]
        verified_files(payload, spec['files']); state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state.update(complete=True, seconds=time.monotonic()-started); save(BASE/'processes.json', state)
        scope['own'].cpu_affinity(scope['prior_affinity'])
    write(BASE/'local-builds.json', state['runs'])
    write(payload/'payload.json', spec)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for name in [*sorted(spec['files']), 'payload.json']: archive.add(payload/name, arcname=name, recursive=False)
    shutil.copy2(PRIOR/'predecessor-failure.json', BASE/'predecessor-failure.json')
    write(BASE/'prepared.json', dict(passed=True, payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz'),
        files=len(spec['files']), bytes=sum(v['bytes'] for v in spec['files'].values()),
        nuget_uncompressed_bytes=prepared['nuget_uncompressed_bytes'], il_bridge=pin(BASE/'logs/local-il-bridge.json'),
        predecessors=dict(prepared['predecessors'], **{(PRIOR/'prepared.json').relative_to(ROOT).as_posix(): pin(PRIOR/'prepared.json')}),
        monitor_sha256=hashlib.sha256(monitor.encode()).hexdigest(), test_harness_adaptation=spec['test_harness_adaptation'],
        scope='One test-only pipe-drain correction, offline CLI/test builds and unchanged Core/Data IL; no model inference or AMD qualification'))
    print(json.dumps(read(BASE/'prepared.json')))


if __name__ == '__main__': main()
