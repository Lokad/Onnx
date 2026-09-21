"""Rebuild only the test harness in a fresh tree, keeping qualified product DLLs."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
import xml.etree.ElementTree as ET
from prepare import ROOT, PAYLOAD, pin, read, save

PRIOR = ROOT/'artifacts/pyannote-lstm-panel-admission-v2-20260921'
FAILED = ROOT/'artifacts/pyannote-lstm-panel-admission-completion-v2-20260921'
BUILT = ROOT/'artifacts/pyannote-lstm-panel-admission-completion-v3-20260921'
BASE = ROOT/'artifacts/pyannote-lstm-panel-admission-completion-v4-20260921'


def main():
    sys.path.insert(0, str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages')); import psutil
    assert pin(FAILED/'failed-preparation.json')['sha256'] == '66cb2567dbc1fe71dab6379527e050dc10b7dee581d72a6de3d1c88967c1607b'
    failure = read(FAILED/'failed-preparation.json')
    for name, wanted in failure['files'].items(): assert pin(FAILED/name) == wanted
    for pid, birth in failure['identities_verified_absent'].items():
        try: assert psutil.Process(int(pid)).create_time() != birth
        except psutil.NoSuchProcess: pass
    original_failure = read(PRIOR/'failed-preparation.json')
    for name, wanted in original_failure['files'].items(): assert pin(PRIOR/name) == wanted
    assert psutil.virtual_memory().available >= 8*1024**3
    assert pin(BUILT/'failed-preparation.json')['sha256'] == '0c49090d7db6e8aaf34e9f6321379041fedfcd73ca6e1f75cd60aaf507a8a2b1'
    built = read(BUILT/'failed-preparation.json')
    for name, wanted in built['files'].items(): assert pin(BUILT/name) == wanted
    source = BUILT/'candidate-source'
    assert pin(source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0/Lokad.Onnx.Backend.Tests.dll') == built['test_assembly']
    BASE.mkdir(); (BASE/'logs').mkdir(); shutil.copy2(Path(__file__), BASE/'resume_cli.py')
    target = Path('tests/Lokad.Onnx.Backend.Tests/CliExitCodeTests.cs')
    before = (PRIOR/'candidate-source'/target).read_text(); after = (ROOT/target).read_text()
    expected = before.replace('        if (!p.WaitForExit(180000))',
        '        // Drain both pipes while the CLI runs; verbose output can exceed their\n'
        '        // capacity before the process reaches its exit-code assertion.\n'
        '        var stdout = p.StandardOutput.ReadToEndAsync();\n'
        '        var stderr = p.StandardError.ReadToEndAsync();\n'
        '        if (!p.WaitForExit(180000))').replace('        return p.ExitCode;',
        '        Task.WhenAll(stdout, stderr).GetAwaiter().GetResult();\n        return p.ExitCode;')
    assert after == expected and after != before
    assert pin(ROOT/target) == pin(source/target)
    binaries = {name: pin(PRIOR/'failed-binaries'/name) for name in ('Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll')}
    tool = PRIOR/'failed-tools/prepare.py'; original = tool.read_text(encoding='utf8')
    start = original.index('    own = psutil.Process();'); end = original.index('    try:\n        backend = Path(', start)
    monitor = textwrap.dedent(original[start:end]); scope = dict(globals(), psutil=psutil)
    exec(compile(monitor, str(tool)+' [preserved monitor]', 'exec'), scope)
    save(BASE/'preparation.json', dict(failed_predecessor=pin(FAILED/'failed-preparation.json'),
        successful_test_build=pin(BUILT/'failed-preparation.json'),
        source_predecessor=pin(PRIOR/'failed-preparation.json'), monitor_sha256=hashlib.sha256(monitor.encode()).hexdigest(),
        test_correction=dict(path=target.as_posix(), before=pin(PRIOR/'candidate-source'/target), after=pin(source/target)),
        binaries=binaries, instructions=pin(PRIOR/'instructions.json')))
    state = scope['state']; started = scope['started']
    try:
        backend = Path('tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj')
        for directory in ('tests/Lokad.Onnx.Backend.Tests', 'src/Lokad.Onnx.CLI'):
            for name, wanted in binaries.items(): assert pin(source/directory/'bin/Release/net10.0'/name) == wanted
        assert pin(source/'tests/Lokad.Onnx.Tensors.Tests/bin/Release/net10.0/Lokad.Onnx.dll') == binaries['Lokad.Onnx.dll']
        scope['test']('candidate-backend', backend, source)
        scope['test']('candidate-tensors', Path('tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'), source)
        ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}; suites = {}
        for name, minimum in [('candidate-backend', 3154), ('candidate-tensors', 342)]:
            path = BASE/'test-results'/(name+'.trx'); tree = ET.parse(path); rows = tree.findall('.//t:UnitTestResult', ns)
            assert all(r.attrib['outcome'] in ('Passed', 'NotExecuted') for r in rows)
            passed = sum(r.attrib['outcome'] == 'Passed' for r in rows); assert passed >= minimum
            suites[name] = dict(passed=passed, skipped=sum(r.attrib['outcome'] == 'NotExecuted' for r in rows), trx=pin(path))
        for name, wanted in original_failure['files'].items(): assert pin(PRIOR/name) == wanted
        for p in source.rglob('*'):
            if p.is_file() and not {'bin', 'obj'}.intersection(p.relative_to(source).parts) and p.relative_to(source) != target:
                assert pin(p) == pin(PRIOR/'candidate-source'/p.relative_to(source)), str(p)
        for directory in ('tests/Lokad.Onnx.Backend.Tests', 'src/Lokad.Onnx.CLI'):
            for name, wanted in binaries.items(): assert pin(source/directory/'bin/Release/net10.0'/name) == wanted
        assert pin(source/'tests/Lokad.Onnx.Tensors.Tests/bin/Release/net10.0/Lokad.Onnx.dll') == binaries['Lokad.Onnx.dll']
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state.update(complete=True, seconds=time.monotonic()-started); save(BASE/'processes.json', state)
        scope['own'].cpu_affinity(scope['prior_affinity'])
    save(BASE/'prepared.json', dict(passed=True, preparation=pin(BASE/'preparation.json'), binaries=binaries,
        suites=suites, focused_tests=114, hardware_disabled_tests=20, instructions=pin(PRIOR/'instructions.json'),
        processes=pin(BASE/'processes.json'), scope='Optional admission and full local suites; test-only output draining; no new model benchmark'))
    print(json.dumps(read(BASE/'prepared.json')))


if __name__ == '__main__': main()
