"""Freeze the prepared M28 source and qualified current control for a Linux build."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-convolution-pointer-build-amd-20260923'
SOURCE = ROOT/'artifacts/pyannote-convolution-pointer-unroll-20260923'
CURRENT = ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
QUALIFIED = ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('build_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)


def previous_closed():
    for folder, digest in [(CURRENT,'6c65419f54f93ac43cc9ca26886dcf4bb9b6535e85f40c4db291fb9b9e1ea4bb'),
                           (QUALIFIED,'5cc03093982964beb44776b7d64cf561e5a6b49c1947b1d6b9a80b790cf41a21')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    assert pin(SOURCE/'prepared.json')['sha256'] == '863f9c85c35c6e8a3610eedcaa1b0014c6a6a6a069fc0ad814b589f1733d10b9'
    source = read(SOURCE/'prepared.json'); assert source['passed'] and not source['root_product_changed']
    for name, wanted in source['files'].items(): assert pin(SOURCE/name) == wanted, name
    for name, wanted in source['tools'].items(): assert pin(ROOT/name) == wanted, name
    for name, wanted in source['before'].items(): assert pin(ROOT/name) == wanted, name
    assert source['changed'] == ['src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs']


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    source = read(SOURCE/'prepared.json')
    for name in source['files']:
        copy(SOURCE/name, bundle/name)
    for p in (QUALIFIED/'bundle/bridge').iterdir():
        if p.is_file(): copy(p, bundle/'bridge'/p.name)
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(), str(p))
        if p.name in ['protocol.py','remote.py','remote_prepare.py','checks.py']: copy(p, bundle/'tools'/p.name)
    for folder, label in [(CURRENT,'current'), (QUALIFIED,'qualified'), (ROOT/'artifacts/pyannote-current-profile-amd-v2-20260922','profile')]:
        copy(folder/'closed.json', bundle/'evidence'/(label+'-closed.json'))
        copy(folder/'analysis.json', bundle/'evidence'/(label+'-analysis.json'))
    copy(SOURCE/'prepared.json', bundle/'evidence/source-prepared.json')
    measured = {name:pin(CURRENT/'collected/runtimes/current'/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    stage = dict(passed=True, measured=measured, source_commit=source['source_commit'],
        source_scope='Isolated M28 row-pointer and fixed-step Kernel512 traversal only; all 416 root files preserved.',
        source_prepared=pin(SOURCE/'prepared.json'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert sum(n.startswith('source/') for n in stage['files']) == 416
    save(bundle/'stage.json', stage)
    for p in [*TOOLS.iterdir(), MONITOR]:
        if p.is_file(): originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'), measured=measured)))


if __name__ == '__main__': prepare()
