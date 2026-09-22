"""Prepare the measured product with a narrowly proved Linux diagnostic adapter."""
import ast
import difflib
import hashlib
import shutil
import tarfile
import traceback
from common import *
import numpy as np


def main():
    assert not BASE.exists()
    assert pin(PRIOR / 'model-closed.json')['sha256'] == '01820a8c5a1ea3afc03e0ef1b555147e8fd0f587d596eec1c8c319258d8cf15e'
    prior = read(PRIOR / 'model-closed.json'); assert prior['passed']; verify(prior['files'])
    for identity in prior['identities']: terminal(identity)
    assert pin(AMD / 'closed.json')['sha256'] == '00ce82bb6e51ff81a28885925b4252bd45de4131caabb254f1fb66780f9bab58'
    amd = read(AMD / 'closed.json'); assert amd['passed']
    archive = read(PAYLOAD / 'payload.json')
    assert pin(PAYLOAD / 'payload.json') == amd['files']['collected/payload.json']
    BASE.mkdir(); (BASE / 'logs').mkdir(); payload = BASE / 'payload'; payload.mkdir()
    files = {rel(PRIOR / 'model-closed.json'):pin(PRIOR / 'model-closed.json'), rel(AMD / 'closed.json'):pin(AMD / 'closed.json')}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target); files[rel(source)] = pin(source)
    for folder in ['runtime', 'tracer']:
        shutil.copytree(PRIOR / folder, payload / folder)
    assert pin(payload / 'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(payload / 'runtime/Lokad.Onnx.Data.dll')['sha256'] == DATA
    source = BASE / 'consumer'; source.mkdir()
    for name in ['Program.cs', 'Diagnostic.cs', 'NpySupport.cs', 'SampledAudio.csproj']:
        copy(PRIOR / 'consumer' / name, source / name)
    p = source / 'Diagnostic.cs'; before = p.read_text(encoding='utf8')
    old = '    [DllImport("kernel32.dll")] internal static extern uint GetCurrentThreadId();'
    new = '''    [DllImport("kernel32.dll", EntryPoint = "GetCurrentThreadId")] static extern uint WindowsThreadId();
    [DllImport("libc", EntryPoint = "gettid")] static extern uint LinuxThreadId();
    internal static uint GetCurrentThreadId() => OperatingSystem.IsLinux() ? LinuxThreadId()
        : OperatingSystem.IsWindows() ? WindowsThreadId() : throw new PlatformNotSupportedException();'''
    assert before.count(old) == 1; after = before.replace(old, new); p.write_text(after, encoding='utf8')
    save(BASE / 'consumer-adaptation.json', dict(reason='Platform thread-ID lookup only',
        diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))))
    bridge = BASE / 'bridge'; bridge.mkdir()
    original = ROOT / 'tests/pyannote/combined-avx512/Inventory.cs.txt'
    code = original.read_text(encoding='utf8')
    old = 'foreach (string name in new[] { "Lokad.Onnx.dll", "Lokad.Onnx.Data.dll" })'
    assert code.count(old) == 1
    (bridge / 'Program.cs').write_text(code.replace(old, 'foreach (string name in new[] { "SampledAudio.dll" })'), encoding='utf8')
    copy(ROOT / 'artifacts/pyannote-portable-integration-20260922/bridge/Bridge.csproj', bridge / 'Bridge.csproj')
    original_pair = ROOT / 'tests/pyannote/sampled-thread-time/common.py'
    pair_before = original_pair.read_text(encoding='utf8').split('def pair(', 1)[1]
    pair_before = 'def pair(' + pair_before; pair = pair_before
    substitutions = [
        ('creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW', 'start_new_session=True', 1),
        ("< .002", "< 1.1", 1),
        ("'10.0.12'", "'10.0.8'", 1),
        ("['dotnet', BASE /", "[DOTNET, BASE /", 1),
        ('20*1024**3', '1024**3', 2),
        ('cpu_system=cpu.system))', 'cpu_system=cpu.system, threads=thread_affinities(p)))', 1),
        ("assert elapsed < 900 and sample['rss']", "assert artifact_size() <= LIMITS['artifacts']\n                assert elapsed < 900 and sample['rss']", 1),
    ]
    for old, new, count in substitutions:
        assert pair.count(old) == count, (old, pair.count(old)); pair = pair.replace(old, new)
    (payload / 'tools').mkdir()
    (payload / 'tools/pair.py.txt').write_text(pair, encoding='utf8'); ast.parse(pair)
    save(BASE / 'process-adaptation.json', dict(source=pin(original_pair), replacements=substitutions,
        diff=''.join(difflib.unified_diff(pair_before.splitlines(True), pair.splitlines(True)))))
    manifest_source = PAYLOAD / 'manifests/portable-pyannote.json'
    assert pin(manifest_source) == archive['files']['manifests/portable-pyannote.json']
    copy(manifest_source, payload / 'manifest.json'); manifest = read(manifest_source)
    external = {}
    entries = [*manifest['models'].values(), *manifest['native_assets'].values(), *manifest['upstream'].values(),
        manifest['reference'], *[c['pcm'] for c in manifest['cases']]]
    for item in entries:
        wanted = {k:item[k] for k in ['bytes', 'sha256']}; name = item['path']
        if name.startswith('/'):
            assert archive['external'][name] == wanted; external[name] = wanted
        else:
            assert pin(PAYLOAD / name) == wanted == archive['files'][name]
            copy(PAYLOAD / name, payload / name)
    expected = AMD / 'collected/campaign/timing-01-portable-output/result.json'
    assert pin(expected) == amd['files'][expected.relative_to(AMD).as_posix()]
    assert read(expected)['core_sha256'] == CORE and read(expected)['data_sha256'] == DATA
    copy(expected, payload / 'prior-amd-result.json')
    raw = {}
    for case in manifest['cases']:
        pcm = np.load(payload / case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        raw[case['name']] = hashlib.sha256(pcm.tobytes()).hexdigest()
    for name, wanted in archive['external'].items():
        if name.startswith('/home/vermorel/.dotnet/') or '/psutil/' in name or name.startswith('/usr/bin/python'):
            external[name] = wanted
    runtime_receipt = ROOT / 'artifacts/pyannote-amd-profile-runtime-20260922.json'
    runtime_inventory = read(runtime_receipt)
    assert runtime_inventory['boot_time'] == 1789634288.0 and not runtime_inventory['workers']
    assert 'Microsoft.NETCore.App 10.0.8' in runtime_inventory['runtimes']
    for name, wanted in runtime_inventory['files'].items():
        assert name not in external or external[name] == wanted
        external[name] = wanted
    assert DOTNET in external
    assert external['/usr/bin/python3'] == archive['interpreter']
    files[rel(runtime_receipt)] = pin(runtime_receipt)
    for name in ['remote.py']:
        copy(TOOLS / name, payload / 'tools' / name)
    copy(ROOT / 'tests/audio/comparison/audit.py', payload / 'tools/public_audit.py')
    for name in ['stacks.py', 'stacks_v2.py']:
        copy(ROOT / 'tests/pyannote/sampled-thread-time' / name, payload / 'tools' / name)
    files[rel(original_pair)] = pin(original_pair)
    for folder in [source, bridge, TOOLS]:
        for path in folder.iterdir():
            if path.is_file(): files[rel(path)] = pin(path)
    save(BASE / 'inputs.json', dict(passed=True, files=files))
    state = new_state(); flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    def run(name, args, output):
        monitor.worker(state, BASE / 'preparation.json', name, args, ROOT, [0], 8, 8, 900, True, output)
        print(name, 'passed', flush=True)
    try:
        for name, project, extra in [('consumer', source / 'SampledAudio.csproj', ['-p:FrozenProductDirectory=' + str(payload / 'runtime')]),
                                      ('bridge', bridge / 'Bridge.csproj', [])]:
            run(name+'-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages', *extra], None)
            run(name+'-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers', *extra], project.parent)
        built = source / 'bin/Release/net10.0'
        run('consumer-instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', PRIOR / 'runtime', built, BASE / 'instructions.json'], bridge)
        inventory = read(BASE / 'instructions.json'); assert inventory['inventory_complete']
        row = inventory['observations'][0]
        assert row['public_surface_equal'] and not row['removed']
        assert len(row['differences']) == 1 and row['differences'][0].startswith('DiagnosticControl::GetCurrentThreadId::')
        assert sorted(k.split('::')[1] for k in row['added']) == ['LinuxThreadId', 'WindowsThreadId']
        for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']: assert pin(built / name) == pin(payload / 'runtime' / name)
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            shutil.copy2(built / ('SampledAudio.'+suffix), payload / 'runtime' / ('SampledAudio.'+suffix))
        verify(files)
        limits = dict(seconds=900, preflight_available=10*1024**3, preflight_tmpfs=3*1024**3, rss=8*1024**3,
            available=1024**3, tmpfs=1024**3, output=1024**3, artifacts=2*1024**3)
        save(payload / 'payload.json', dict(files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()},
            external=external, core=CORE, data=DATA, consumer=pin(payload / 'runtime/SampledAudio.dll'), raw_pcm=raw,
            boot_time=1789634288.0, limits=limits, jobs=['control','sampled-a','sampled-b']))
        with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
            for p in sorted(payload.rglob('*')):
                if p.is_file(): tar.add(p, arcname=p.relative_to(payload).as_posix(), recursive=False)
        files.update({rel(p):pin(p) for p in payload.rglob('*') if p.is_file()})
        save(BASE / 'prepared.json', dict(passed=True, files=files, archive=pin(BASE / 'payload.tar.gz'), payload=pin(payload / 'payload.json'),
            instructions=pin(BASE / 'instructions.json'), existing_methods=row['methods'], unchanged_methods=row['unchanged_methods']))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'preparation.json', state)


if __name__ == '__main__': main()
