"""Freeze the current measured product, closed qualification and baseline tools."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]; TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
APP = ROOT/'artifacts/pyannote-winograd-product-app-amd-20260923'
PRIOR = {
    'application': (APP, 'dc2c7b9f5086ab9b4ee615b9dad7643eaf4c1ed65c1cc71b3e76ee794237d88e'),
    'root': (ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923', '62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'),
    'parakeet': (ROOT/'artifacts/pyannote-winograd-product-parakeet-amd-20260923', '38b346aec0df29b4e99163282253012a5f066b2d80615cd7517a730b199ac14f')}
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
