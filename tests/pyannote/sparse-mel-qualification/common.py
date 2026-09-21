"""Frozen Data-only application qualification and inherited Core evidence."""
import difflib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
MODEL = ROOT / 'artifacts/pyannote-sparse-mel-20260921'
FRONTEND = ROOT / 'artifacts/pyannote-sparse-mel-frontend-20260921'
BASE = ROOT / 'artifacts/pyannote-sparse-mel-applications-20260921'
PRIOR = ROOT / 'artifacts/pyannote-convolution-portable-applications-20260921'
PUBLIC = ROOT / 'artifacts/pyannote-optimized-ort-20260921'
MEETINGS = ROOT / 'artifacts/pyannote-optimized-meetings-20260921'
INPUT = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
CORE = '5c0ae2aa7c3cce58f3ffcb190df451e053a449a3e0dbc920d7b6d0b2bc66020c'
DATA = 'e9e4c28e2f7277ea226556f692d95de3d9d4a5f94eed79a9235268137dcd4775'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('sparse_application_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path):
    return path.relative_to(ROOT).as_posix()


def prerequisites():
    receipts = [
        (MODEL / 'focused-closed.json', '35cba7a867ee2faff0fb8c3b94431c6f3ffa390648242188da2cd6515856cb6b'),
        (FRONTEND / 'closed.json', '71ac5283b789ff9ea4c8e55f18b12c54352656b020b323eb75dc4fc63a86a40d'),
        (PRIOR / 'closed.json', '20fd9f38b80f2f1e2b87a375b0b2c1c99e255d77f69b674271614361e7282002'),
        (ROOT / 'artifacts/pyannote-convolution-portable-shared-20260921/closed.json', '32f84f780f9fbb74080ee7414a1e477f92bb9cd1fe287620ed50b7e833bb4282'),
        (ROOT / 'artifacts/pyannote-convolution-portable-parakeet-20260921/closed.json', '56ceec578ca673b36ab685c456c248d8fd09810fc561c69f7ccc24dc72fa2751'),
    ]
    files = {}
    for path, sha in receipts:
        assert pin(path)['sha256'] == sha
        closure = read(path)
        assert closure['passed']
        verify(closure['files'])
        files.update(closure['files'])
        files[rel(path)] = pin(path)
        for identity in closure.get('identities', closure.get('terminal_identities', [])):
            terminal(identity)
    assert pin(MODEL / 'runtime/Lokad.Onnx.dll') == pin(PRIOR / 'application-runtime/Lokad.Onnx.dll')
    assert pin(MODEL / 'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(MODEL / 'runtime/Lokad.Onnx.Data.dll')['sha256'] == DATA
    assert read(MODEL / 'instructions.json')['passed']
    for path in [MONITOR, *TOOLS.glob('*.py'), ROOT / 'tests/pyannote/request-contexts/qualify.py',
                 ROOT / 'tests/pyannote/request-contexts/audit.py', ROOT / 'tests/Lokad.Onnx.Backend.Tests/CliExitCodeTests.cs']:
        files[rel(path)] = pin(path)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    save(BASE / 'inherited-core-evidence.json', dict(core=CORE, data=DATA,
        receipts=[dict(path=rel(path), sha256=sha) for path, sha in receipts],
        scope='Core is byte-identical; inherit its 166 shared-model arrays and 784 Parakeet arrays, retaining the three known native duration failures. All 694 other Data methods are unchanged. Run complete affected pyannote consumers anew.'))
    files[rel(BASE / 'inherited-core-evidence.json')] = pin(BASE / 'inherited-core-evidence.json')
    return dict(passed=True, files=files)


def correct_cli(source):
    relative = 'tests/Lokad.Onnx.Backend.Tests/CliExitCodeTests.cs'
    path = source / relative
    before = path.read_text(encoding='utf8')
    assert before.count('        if (!p.WaitForExit(180000))') == 1 and before.count('        return p.ExitCode;') == 1
    after = before.replace('        if (!p.WaitForExit(180000))',
        '        // Drain both pipes while the CLI runs; verbose output can exceed their\n'
        '        // capacity before the process reaches its exit-code assertion.\n'
        '        var stdout = p.StandardOutput.ReadToEndAsync();\n'
        '        var stderr = p.StandardError.ReadToEndAsync();\n'
        '        if (!p.WaitForExit(180000))').replace('        return p.ExitCode;',
        '        Task.WhenAll(stdout, stderr).GetAwaiter().GetResult();\n        return p.ExitCode;')
    assert after == (ROOT / relative).read_text(encoding='utf8')
    path.write_text(after, encoding='utf8')
    save(BASE / 'cli-test-correction.json', dict(path=relative, original=pin(MODEL / 'source' / relative),
        corrected=pin(path), existing_root_correction=pin(ROOT / relative),
        diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True))),
        scope='Drain existing redirected test pipes; unchanged CLI, timeout and exit-code assertions.'))
