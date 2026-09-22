"""Freeze unchanged Parakeet consumers and the closed Pyannote-qualified product."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-lstm-input-parakeet-amd-20260922'
MODELS = ROOT/'artifacts/pyannote-lstm-input-models-amd-20260922'
APP = ROOT/'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922'
APP_PAYLOAD = ROOT/'artifacts/pyannote-blocked-spatial-app-amd-payload-20260922/payload'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('parakeet_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)


def previous_closed():
    for folder, digest in [
        (MODELS, '2a348f67238a44e3cc36f58e8cd5bf467fa1a8562c57c788a04902f8bdb8c90b'),
        (APP, '5c238cd33845eb58fc00332361a967185ae82a9fcb70854530e07fad58f064d0')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        for name, wanted in read(folder/'closed.json')['files'].items():
            assert pin(folder/name) == wanted, name
    assert pin(APP_PAYLOAD/'payload.json')['sha256'] == '229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f'


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir()
    bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'native_audit.py', 'public_audit.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    for name, source in [('native_audit.py', ROOT/'tests/parakeet/transcribe/audit.py'), ('public_audit.py', ROOT/'tests/audio/comparison/audit.py')]:
        assert (TOOLS/name).read_bytes() == source.read_bytes(); originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['closed.json', 'analysis.json']:
        copy(MODELS/name, bundle/'evidence'/('models-'+name))
    copy(APP_PAYLOAD/'manifests/portable-parakeet.json', bundle/'evidence/original-manifest.json')
    copy(ROOT/'PLAN.md', bundle/'prospective-plan.md'); originals.pop('PLAN.md')
    original = APP_PAYLOAD/'manifests/portable-parakeet.json'
    assert pin(original) == read(APP_PAYLOAD/'payload.json')['files']['manifests/portable-parakeet.json']
    stage = dict(passed=True, identities=read(MODELS/'analysis.json')['identities'],
        consumers={name: read(APP_PAYLOAD/'payload.json')['files']['runtimes/portable/'+name+'.dll'] for name in ['TranscribeReplay', 'AudioBenchmark']},
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert stage['consumers']['TranscribeReplay']['sha256'] == '335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
    assert stage['consumers']['AudioBenchmark']['sha256'] == '7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    save(bundle/'stage.json', stage)
    files = dict(originals)
    for p in [*TOOLS.iterdir(), MONITOR, MODELS/'closed.json', APP/'closed.json', APP_PAYLOAD/'payload.json']:
        if p.is_file(): files[p.relative_to(ROOT).as_posix()] = pin(p)
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=files, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'))))


if __name__ == '__main__': prepare()
