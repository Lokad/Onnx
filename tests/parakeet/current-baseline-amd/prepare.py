"""Freeze the current measured product, closed qualification and baseline tools."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]; TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
APP = ROOT/'artifacts/pyannote-lstm-input-app-amd-20260922'
PRIOR = {
    'application': (APP, '73a4897a4db8e1bd729cb3c5486bcb11814d4ff669c9b9a472003572b08c64d0'),
    'root': (ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922', '5cc03093982964beb44776b7d64cf561e5a6b49c1947b1d6b9a80b790cf41a21'),
    'parakeet': (ROOT/'artifacts/pyannote-lstm-input-parakeet-amd-20260922', '36c475cfde426125dc58df4f40f14318d5e55fb8bc6c9b68b225c7885706d17f')}
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
module = importlib.util.spec_from_file_location('baseline_monitor', MONITOR)
monitor = importlib.util.module_from_spec(module); module.loader.exec_module(monitor)


def previous_closed():
    for folder, digest in PRIOR.values():
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    assert read(APP/'analysis.json')['performance']['admitted']


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir()
    bundle = BASE/'bundle'; bundle.mkdir(); originals = {}; prerequisites = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'statistics_exact.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    for name, (folder, digest) in PRIOR.items():
        for filename in ['closed.json', 'analysis.json']: copy(folder/filename, bundle/'evidence'/name/filename)
        prerequisites[name] = dict(closed=pin(folder/'closed.json'), analysis=pin(folder/'analysis.json'))
    copy(PRIOR['parakeet'][0]/'collected/candidate-public/output/result.json', bundle/'evidence/current-public.json')
    copy(APP/'collected/manifests/candidate-parakeet.json', bundle/'evidence/original-parakeet.json')
    copy(ROOT/'PLAN.md', bundle/'prospective-plan.md'); originals.pop('PLAN.md')
    app = read(APP/'analysis.json')
    stage = dict(passed=True, identities=dict(current=app['identities']['candidate']), prerequisites=prerequisites,
        consumers=dict(AudioBenchmark=app['consumers']['AudioBenchmark']),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    files = dict(originals)
    for p in [*TOOLS.iterdir(), MONITOR, APP/'payload.json']:
        if p.is_file(): files[p.relative_to(ROOT).as_posix()] = pin(p)
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=files, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'))))


if __name__ == '__main__': prepare()
