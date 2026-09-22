"""Pin complete audio inputs and correct only the Pyannote consumer Data literal."""
import shutil
import traceback
from common import *

OLD_CONSUMER = ROOT / 'artifacts/pyannote-combined-consumers-20260922'
OLD_DATA = '85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'


def main():
    candidate()
    assert not BASE.exists()
    assert pin(OLD_CONSUMER / 'closed.json')['sha256'] == '602d752956d6cf830851cd0dc35eb4bdf1230880fb2dce6d8bbea6f530bb555c'
    prior = read(OLD_CONSUMER / 'closed.json')
    assert prior['passed']
    verify(prior['files'])
    for identity in prior['identities']:
        terminal(identity)
    assert pin(SELECTED / 'closed.json')['sha256'] == '56907aea97ef3af546336e61ef713c9ce5d8179d847a8363bbabfdb1463b531c'
    selected = read(SELECTED / 'closed.json'); assert selected['passed']; verify(selected['files'])
    for identity in selected['terminal_identities']: terminal(identity)
    BASE.mkdir(); (BASE / 'logs').mkdir()
    runtime = BASE / 'runtime'
    shutil.copytree(MODEL / 'runtime', runtime)
    consumer = BASE / 'consumer'; consumer.mkdir()
    for name in ['Program.cs', 'NpySupport.cs', 'GraphQualification.csproj']:
        shutil.copy2(OLD_CONSUMER / 'portable' / name, consumer / name)
    p = consumer / 'Program.cs'; before = p.read_text(encoding='utf8')
    assert before.count(OLD_DATA) == 1
    p.write_text(before.replace(OLD_DATA, DATA), encoding='utf8')
    assert p.read_text(encoding='utf8').replace(DATA, OLD_DATA) == before
    for source, name in [(ROOT / 'artifacts/parakeet-transcription-20260919/frozen/replay', 'TranscribeReplay'),
                         (ROOT / 'artifacts/audio-ort-baseline-v2-20260919/bin', 'AudioBenchmark')]:
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            shutil.copy2(source / (name + '.' + suffix), runtime / (name + '.' + suffix))
    assert pin(runtime / 'TranscribeReplay.dll')['sha256'] == '335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
    assert pin(runtime / 'AudioBenchmark.dll')['sha256'] == '7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    files = {}
    def bind(p): files[rel(p)] = pin(p)
    for folder in [runtime, consumer, TOOLS]:
        for p in folder.iterdir():
            if p.is_file(): bind(p)
    for p in [MODEL / 'closed.json', MODEL / 'external-operand-proof.json', OLD_CONSUMER / 'closed.json', MONITOR,
              ROOT / 'tests/parakeet/transcribe/audit.py', ROOT / 'tests/audio/comparison/audit.py',
              ROOT / 'tests/parakeet/reduction-shared/qualify_pyannote.py', REFERENCE, CORPUS, INPUT,
              ROOT / 'artifacts/pyannote-performance-profile-20260921/output/result.json']:
        bind(p)
    for p in [SELECTED / 'closed.json', SELECTED / 'output/result.json']: bind(p)
    # Native tensors, complete models, corpus audio, and all referenced assets.
    for name, wanted in read(REFERENCE)['files'].items():
        p = REFERENCE.parent / name
        assert pin(p) == {k: wanted[k] for k in ['bytes', 'sha256']}; bind(p)
    for manifest in [read(CORPUS), read(INPUT)]:
        specs = list(manifest['models'].values()) + [manifest['reference']] + [c['pcm'] for c in manifest['cases']]
        specs += list(manifest.get('native_assets', {}).values())
        upstream = manifest['upstream']; specs += [upstream] if 'path' in upstream else list(upstream.values())
        for item in specs:
            p = ROOT / item['path']; assert pin(p) == {k: item[k] for k in ['bytes', 'sha256']}; bind(p)
    reference = ROOT / read(INPUT)['reference']['path']
    for name, item in read(reference)['files'].items():
        p = reference.parent / name; assert pin(p) == {k: item[k] for k in ['bytes', 'sha256']}; bind(p)
    bridge = OLD_CONSUMER / 'bridge/bin/Release/net10.0'
    for p in bridge.iterdir():
        if p.is_file(): bind(p)
    save(BASE / 'inputs.json', dict(passed=True, files=files))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    monitor.BASE = BASE
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    def run(name, args):
        monitor.worker(state, BASE / 'build-state.json', name, args, ROOT, [0], 8, 8, 900, True, consumer)
        print(name, 'passed', flush=True)
    try:
        project = consumer / 'GraphQualification.csproj'
        extra = ['-p:FrozenProductDirectory=' + str(runtime)]
        run('consumer-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages', *extra])
        run('consumer-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers', *extra])
        built = consumer / 'bin/Release/net10.0'
        run('consumer-instructions', ['dotnet', bridge / 'Bridge.dll', OLD_CONSUMER / 'portable/bin/Release/net10.0', built, BASE / 'consumer-instructions.json'])
        inventory = read(BASE / 'consumer-instructions.json')
        assert inventory['inventory_complete'] and len(inventory['observations']) == 1
        row = inventory['observations'][0]
        assert row['methods'] == 96 and row['public_surface_equal'] and not row['added'] and not row['removed']
        assert len(row['differences']) == 1
        method = row['differences'][0]; assert method.startswith('Program::<Main>$::')
        old, new = row['normalized_methods'][method], row['candidate_methods'][method]
        assert old.count(OLD_DATA) == 1 and old.replace(OLD_DATA, DATA) == new
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            shutil.copy2(built / ('GraphQualification.' + suffix), runtime / ('GraphQualification.' + suffix))
        assert pin(runtime / 'Lokad.Onnx.dll')['sha256'] == CORE and pin(runtime / 'Lokad.Onnx.Data.dll')['sha256'] == DATA
        verify(files)
        files.update({rel(p): pin(p) for p in runtime.iterdir() if p.is_file()})
        save(BASE / 'prepared.json', dict(passed=True, core=CORE, data=DATA, files=files,
            consumer_methods=96, unchanged_consumer_methods=95, changed_literal=dict(before=OLD_DATA, after=DATA),
            jobs=dict(native=[10, 8, 1200, True], public=[14, 12, 1200, True], pyannote=[10, 8, 900, True]),
            scope='Complete native/public correctness only; no performance or production promotion'))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'build-state.json', state)


if __name__ == '__main__': main()
