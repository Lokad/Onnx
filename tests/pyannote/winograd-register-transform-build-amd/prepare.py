"""Freeze the isolated one-method source and current measured product."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-winograd-register-transform-build-amd-20260923'
SOURCE = ROOT/'artifacts/pyannote-winograd-register-transform-source-20260923'
CURRENT = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
LATEST = ROOT/'artifacts/release-graph-baseline-amd-v2-20260923'
QUALIFIED = ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('build_monitor',MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)


def previous_closed():
    for folder,digest in [(LATEST,'7bfcfa23a9dcc3aa2b6bff22acf612fd140a82bff62a5d7bf891d92da420be9a'),(CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'),
                          (QUALIFIED,'62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name,wanted in proof['files'].items(): assert pin(folder/name) == wanted,name
    assert pin(SOURCE/'prepared.json')['sha256'] == 'd99934ab4ce2715cf48b7de61f5760a8d5c52a37adb617c673f133822fe3242a'
    source = read(SOURCE/'prepared.json')
    assert source['passed'] and not source['root_product_changed'] and not source['built']
    assert len(source['source']) == len(source['before']) == 420
    for name,wanted in source['source'].items(): assert pin(SOURCE/'source'/name) == wanted,name
    for name,wanted in source['before'].items(): assert pin(ROOT/name) == wanted,name
    assert source['changed'] == ['src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Winograd.cs']
    assert pin(SOURCE/'candidate.patch') == source['patch'] and pin(SOURCE/'prospective-plan.md') == source['plan']
    assert pin(ROOT/'tests/pyannote/winograd-register-transform-source/prepare.py') == source['generator']


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    source = read(SOURCE/'prepared.json')
    for name in source['source']: copy(SOURCE/'source'/name,bundle/'source'/name)
    for p in (QUALIFIED/'bundle/bridge').iterdir():
        if p.is_file(): copy(p,bundle/'bridge'/p.name)
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(),str(p))
        if p.name in ['protocol.py','remote.py','remote_prepare.py','checks.py']: copy(p,bundle/'tools'/p.name)
    for folder,label in [(CURRENT,'current'),(QUALIFIED,'qualified')]:
        for name in ['closed.json','analysis.json','payload.json']:
            copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    copy(LATEST/'collected/collection.json',bundle/'evidence/latest-collection.json')
    copy(LATEST/'closed.json',bundle/'evidence/latest-closed.json')
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(SOURCE/'candidate.patch',bundle/'candidate.patch')
    shutil.copy2(ROOT/'.agent/m37-winograd-register-transform-20260923.md',bundle/'prospective-plan.md')
    measured = {name:pin(CURRENT/'collected/runtimes/current'/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    stage = dict(passed=True,measured=measured,source_commit='94a550de',source_prepared=pin(SOURCE/'prepared.json'),
        source_scope='Only TransformWinogradInputContiguous changes in an isolated copy; all420rootfiles preserved.',
        measured_files={p.name:pin(p) for p in (CURRENT/'collected/runtimes/current').iterdir() if p.is_file()},
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert sum(n.startswith('source/') for n in stage['files']) == 420
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file(): originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),measured=measured)))


if __name__ == '__main__': prepare()
