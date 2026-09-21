"""Change only the diagnostic consumer's required product hash literals."""
import difflib
from common import *


def main():
    assert not BASE.exists()
    files = {}
    for path, sha in [(PRIOR / 'model-closed.json', '71c2e9c36f31216fe538454426aea9a15d7beaa7a19f72c038aaa7fec6bf818f'),
        (QUALIFIED / 'closed.json', '9d62d3d21a174e9be7105e4224b4b4b986b89dc6085ec60cc99b3aa2fdd7da77')]:
        assert pin(path)['sha256'] == sha
        proof = read(path)
        assert proof['passed']
        verify_spec(proof)
        for identity in proof['identities']:
            terminal(identity)
        files[rel(path)] = pin(path)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    runtime, source = BASE / 'runtime', BASE / 'consumer'
    shutil.copytree(QUALIFIED / 'runtime', runtime)
    shutil.copytree(PRIOR / 'tracer', BASE / 'tracer')
    source.mkdir()
    for name in ['Program.cs', 'Diagnostic.cs', 'NpySupport.cs', 'SampledAudio.csproj']:
        shutil.copy2(PRIOR / 'consumer' / name, source / name)
    before = (source / 'Program.cs').read_text(encoding='utf8')
    assert before.count(OLD_CORE) == before.count(OLD_DATA) == 1
    after = before.replace(OLD_CORE, CORE).replace(OLD_DATA, DATA)
    (source / 'Program.cs').write_text(after, encoding='utf8')
    save(BASE / 'consumer-adaptation.json', dict(original=pin(PRIOR / 'consumer/Program.cs'), candidate=pin(source / 'Program.cs'),
        diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True))),
        scope='Only two required product digest literals change; all diagnostic/native/public/ownership source assertions are identical.'))
    assert pin(runtime / 'Lokad.Onnx.dll')['sha256'] == CORE and pin(runtime / 'Lokad.Onnx.Data.dll')['sha256'] == DATA
    for path in [*TOOLS.iterdir(), *OLD_TOOLS.glob('*.py'), ROOT / 'tests/audio/comparison/audit.py',
        QUALIFIED / 'dialogue-output/result.json', INPUT, BASE / 'consumer-adaptation.json']:
        if path.is_file():
            files[rel(path)] = pin(path)
    manifest = read(INPUT)
    for item in [*manifest['models'].values(), *manifest['native_assets'].values(), *manifest['upstream'].values(),
        manifest['reference'], *[case['pcm'] for case in manifest['cases']]]:
        path = ROOT / item['path']
        assert pin(path) == {key: item[key] for key in ['bytes', 'sha256']}
        files[rel(path)] = pin(path)
    for folder in [source, runtime, BASE / 'tracer']:
        for path in folder.rglob('*'):
            if path.is_file():
                files[rel(path)] = pin(path)
    save(BASE / 'source-prepared.json', dict(passed=True, files=files))
    state = new_state()
    save(BASE / 'model-preparation.json', state)
    try:
        flags = monitor.FLAGS + ['-p:NuGetAudit=false', '-p:FrozenProductDirectory=' + str(runtime)]
        for name, command in [('consumer-restore', ['dotnet', 'restore', source / 'SampledAudio.csproj', *flags, '--source', FEED, '--packages', BASE / 'packages']),
            ('consumer-build', ['dotnet', 'build', source / 'SampledAudio.csproj', '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])]:
            monitor.worker(state, BASE / 'model-preparation.json', name, command, ROOT, [0], 8, 8, 900, True, source)
            print(name, 'passed', flush=True)
        binary = source / 'bin/Release/net10.0'
        for name in ['Lokad.Onnx', 'Lokad.Onnx.Data', 'Google.Protobuf', 'FastBertTokenizer', 'Lokad.Tokenizers', 'SixLabors.ImageSharp']:
            assert pin(binary / (name + '.dll')) == pin(runtime / (name + '.dll'))
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            shutil.copy2(binary / ('SampledAudio.' + suffix), runtime / ('SampledAudio.' + suffix))
        verify(files)
        for folder in [source, runtime]:
            for path in folder.rglob('*'):
                if path.is_file() and 'obj' not in path.relative_to(folder).parts:
                    files[rel(path)] = pin(path)
        save(BASE / 'model-prepared.json', dict(passed=True, files=files, external_files={}, core=CORE, data=DATA,
            consumer=pin(runtime / 'SampledAudio.dll'), jobs=['control', 'sampled-a', 'sampled-b'],
            limits=dict(preflight_gib=10, aggregate_rss_gib=8, available_gib=1, disk_gib=20, output_gib=1, seconds=900),
            scope='Current integrated product attribution with original complete public diagnostic protocol; no matched speed or AMD claim.'))
        state['code'] = 0
        print(dict(prepared=pin(BASE / 'model-prepared.json')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'model-preparation.json', state)


if __name__ == '__main__':
    main()
