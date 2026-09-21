"""Freeze real PCM and exact Data identities before a frontend-only replay."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-sparse-mel-frontend-20260921'
MODEL = ROOT / 'artifacts/pyannote-sparse-mel-20260921'
PRIOR = ROOT / 'artifacts/pyannote-convolution-portable-applications-20260921'
INPUT = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('sparse_frontend_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    assert not BASE.exists()
    receipt = MODEL / 'focused-closed.json'
    assert pin(receipt)['sha256'] == '35cba7a867ee2faff0fb8c3b94431c6f3ffa390648242188da2cd6515856cb6b'
    closed = read(receipt)
    assert closed['passed']
    verify(closed['files'])
    for identity in closed['identities']:
        terminal(identity)
    prior = read(PRIOR / 'closed.json')
    assert pin(INPUT) == prior['files'][rel(INPUT)]
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    (BASE / 'inputs').mkdir()
    import numpy as np
    files = {rel(receipt): pin(receipt), rel(INPUT): pin(INPUT), rel(MONITOR): pin(MONITOR)}
    cases = []
    for case in read(INPUT)['cases']:
        path = ROOT / case['pcm']['path']
        assert pin(path) == {k: case['pcm'][k] for k in ['bytes', 'sha256']}
        pcm = np.load(path, allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.all(np.isfinite(pcm)) and np.all(np.abs(pcm) <= 1)
        raw = BASE / 'inputs' / (case['name'] + '.f32')
        raw.write_bytes(pcm.tobytes())
        files[rel(path)], files[rel(raw)] = pin(path), pin(raw)
        row = dict(name=case['name'], path=rel(raw), offset=0, samples=case['samples'], **pin(raw))
        cases.append(row)
    full = cases[0]
    assert full['samples'] == 480000
    for start in range(21):
        cases.append(dict(full, name='full-window-' + str(start), offset=start * 16000, samples=160000))
    assert len(cases) == 25
    baseline = PRIOR / 'application-runtime/Lokad.Onnx.Data.dll'
    assert pin(baseline)['sha256'] == '1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662'
    files[rel(baseline)] = pin(baseline)
    manifest = dict(core=pin(MODEL / 'runtime/Lokad.Onnx.dll')['sha256'], data=pin(MODEL / 'runtime/Lokad.Onnx.Data.dll')['sha256'],
        baseline_data_path=rel(baseline), cases=cases)
    save(BASE / 'manifest.json', manifest)
    source = BASE / 'consumer'
    source.mkdir()
    shutil.copy2(TOOLS / 'Probe.cs', source / 'Program.cs')
    (source / 'Probe.csproj').write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup>
<Reference Include="Lokad.Onnx"><HintPath>$(FrozenProductDirectory)/Lokad.Onnx.dll</HintPath></Reference>
<Reference Include="Lokad.Onnx.Data"><HintPath>$(FrozenProductDirectory)/Lokad.Onnx.Data.dll</HintPath></Reference>
</ItemGroup></Project>\n''', encoding='utf8')
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    save(BASE / 'preparation.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false', '-p:FrozenProductDirectory=' + str(MODEL / 'runtime')]
    try:
        for name, args in [('restore', ['dotnet', 'restore', source / 'Probe.csproj', *flags, '--source', FEED, '--packages', BASE / 'packages']),
                           ('build', ['dotnet', 'build', source / 'Probe.csproj', '-c', 'Release', *flags, '--no-restore', '--disable-build-servers', '-o', BASE / 'runtime'])]:
            monitor.worker(state, BASE / 'preparation.json', name, args, source, [0], 8, 4, 900, True, None)
        for path in (MODEL / 'runtime').glob('*.dll'):
            target = BASE / 'runtime' / path.name
            if not target.exists():
                shutil.copy2(path, target)
            assert pin(target) == pin(path)
        for folder in [TOOLS, source, BASE / 'runtime']:
            for path in folder.rglob('*'):
                if path.is_file() and not {'obj', 'bin'}.intersection(path.relative_to(folder).parts):
                    files[rel(path)] = pin(path)
        files[rel(BASE / 'manifest.json')] = pin(BASE / 'manifest.json')
        save(BASE / 'prepared.json', dict(passed=True, files=files, manifest=manifest, jobs=['dense-first', 'sparse-first'],
            expected_pairs=50, expected_values=4312000, limits=dict(preflight_gib=10, rss_gib=4, seconds=900)))
        state['code'] = 0
        print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'))))
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'preparation.json', state)


if __name__ == '__main__':
    main()
