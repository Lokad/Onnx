"""Complete full suites after building the omitted CLI; retain earlier evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
import xml.etree.ElementTree as ET
from prepare import ROOT, PAYLOAD, pin, read, save

PRIOR = ROOT/'artifacts/pyannote-lstm-panel-admission-v2-20260921'
BASE = ROOT/'artifacts/pyannote-lstm-panel-admission-completion-v2-20260921'


def main():
    sys.path.insert(0, str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    refused = ROOT/'artifacts/pyannote-lstm-panel-admission-completion-20260921'
    assert pin(refused/'failed-preparation.json')['sha256'] == '05dfdffac3edd9da6d3331a34b3c94e1a29e2716855ec263678509aadd072287'
    for name, wanted in read(refused/'failed-preparation.json')['files'].items(): assert pin(refused/name) == wanted
    assert read(refused/'processes.json')['runs'] == []
    assert psutil.virtual_memory().available >= 8*1024**3, 'No artifact or child created before sufficient memory'
    failure_path = PRIOR/'failed-preparation.json'
    assert pin(failure_path)['sha256'] == '8b60c89ee17ca0425f848b0b616644cdd00c28dda12f31ef04de741d27f6f805'
    failure = read(failure_path)
    assert not failure['passed'] and failure['backend_counts']['failed'] == '51'
    for name, wanted in failure['files'].items(): assert pin(PRIOR/name) == wanted, name
    for pid, birth in failure['identities_verified_absent'].items():
        try: assert psutil.Process(int(pid)).create_time() != birth
        except psutil.NoSuchProcess: pass
    source = PRIOR/'candidate-source'
    binary_dir = source/'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'
    binaries = {p.name: pin(p) for p in (PRIOR/'failed-binaries').iterdir()}
    for name, wanted in binaries.items(): assert pin(binary_dir/name) == wanted
    instructions = read(PRIOR/'instructions.json'); assert instructions['passed']
    assert [(r['assembly'], r['unchanged_methods'], len(r['added_methods']), len(r['changed_methods']))
            for r in instructions['observations']] == [('Lokad.Onnx.dll', 3103, 1, 1), ('Lokad.Onnx.Data.dll', 691, 0, 0)]
    BASE.mkdir(); (BASE/'logs').mkdir()
    shutil.copy2(Path(__file__), BASE/'resume.py')
    # Reuse the exact bounded process monitor from the preserved failed tool.
    # No repeat of focused tests, negative control or instruction comparison.
    tool = PRIOR/'failed-tools/prepare.py'; original = tool.read_text(encoding='utf8')
    start = original.index('    own = psutil.Process();')
    end = original.index('    try:\n        backend = Path(', start)
    monitor = textwrap.dedent(original[start:end]).replace("BASE/'packages'", "PRIOR/'packages'")
    scope = dict(globals(), psutil=psutil)
    exec(compile(monitor, str(tool)+' [preserved monitor]', 'exec'), scope)
    save(BASE/'preparation.json', dict(failed_predecessor=pin(failure_path), binaries=binaries,
        preflight_refusal=pin(refused/'failed-preparation.json'),
        source=pin(tool), monitor_sha256=hashlib.sha256(monitor.encode()).hexdigest(),
        instructions=pin(PRIOR/'instructions.json'), scope='Full suites with CLI prerequisite; no new model benchmark'))
    state = scope['state']; started = scope['started']
    try:
        build = scope['restore_build']; test = scope['test']
        build('cli', Path('src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'), source)
        tensors = Path('tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')
        build('tensors', tensors, source)
        for name in ('Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'):
            assert pin(source/'src/Lokad.Onnx.CLI/bin/Release/net10.0'/name) == binaries[name]
        for name, wanted in binaries.items(): assert pin(binary_dir/name) == wanted
        test('candidate-backend', Path('tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'), source)
        test('candidate-tensors', tensors, source)
        for name, wanted in failure['files'].items(): assert pin(PRIOR/name) == wanted, name
        for name, wanted in binaries.items(): assert pin(binary_dir/name) == wanted
        ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}; suites = {}
        for name, minimum in [('candidate-backend', 3148), ('candidate-tensors', 342)]:
            path = BASE/'test-results'/(name+'.trx'); tree = ET.parse(path)
            rows = tree.findall('.//t:UnitTestResult', ns)
            assert all(r.attrib['outcome'] in ('Passed', 'NotExecuted') for r in rows)
            passed = sum(r.attrib['outcome'] == 'Passed' for r in rows); assert passed >= minimum
            suites[name] = dict(passed=passed, skipped=sum(r.attrib['outcome'] == 'NotExecuted' for r in rows), trx=pin(path))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state.update(complete=True, seconds=time.monotonic()-started); save(BASE/'processes.json', state)
        scope['own'].cpu_affinity(scope['prior_affinity'])
    save(BASE/'prepared.json', dict(passed=True, predecessor=pin(failure_path), binaries=binaries,
        instructions=pin(PRIOR/'instructions.json'), suites=suites,
        focused_tests=114, hardware_disabled_tests=20, processes=pin(BASE/'processes.json'),
        scope='Optional admission and full local suites only; AMD row execution and model timing remain pending'))
    print(json.dumps(read(BASE/'prepared.json')))


if __name__ == '__main__':
    main()
