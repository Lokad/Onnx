"""Freeze actual graph residency at unchanged 256/64 MiB budgets."""
import ast
import json
from pathlib import Path
import re
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-inclusive-packing-residency-amd-20260924'
BUILD = ROOT / 'artifacts/parakeet-inclusive-packing-build-amd-20260924'
SOURCE = ROOT / 'artifacts/parakeet-inclusive-packing-source-20260924'
PARENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
CONTRACTS = ROOT / 'artifacts/parakeet-inclusive-packing-contracts-amd-v2-20260924'
MODELS = ROOT / 'artifacts/parakeet-wide-entry-first-use-models-amd-20260923'


def previous_closed():
    for folder, digest in [(BUILD, '50f2a3a8a2ebbe20d41315bc0be4242e24fb7f3eb74e78533091a750791a4078'),
                           (PARENT, 'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'),
                           (CONTRACTS, 'e85e5f5f7d435141b41b19957cc9bc119850c0593a120956002c798e66d01b76'),
                           (MODELS, 'f30100534cbb79db790aac30365d533e6d3dcce79e779abc5feb8f7cf3fc1e22')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    assert pin(SOURCE / 'prepared.json')['sha256'] == 'a535266a48631edc8578eeab679df644884d08f1f1cb08e84dbb27415b1cbd7c'
    source = read(SOURCE / 'prepared.json')
    for name, wanted in source['source'].items(): assert pin(SOURCE / 'source' / name) == wanted, name
    for name, wanted in source['before'].items(): assert pin(ROOT / name) == wanted, name
    assert read(BUILD / 'analysis.json')['inventory']['opcode_after'] == 'bgt.s'


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    copy(SOURCE / 'source/global.json', bundle / 'source/global.json')
    for name in ['Program.cs', 'Census.csproj']: copy(TOOLS / name, bundle / 'source' / name)
    manifest = MODELS / 'collected/manifests/candidate-parakeet.json'
    copy(manifest, bundle / 'evidence/model-manifest.json')
    model_spec = read(manifest)
    names = ['encoder-model.onnx', 'encoder-model.onnx.data', 'decoder_joint-model.onnx']
    models = {graph: model_spec['models'][model_spec['graphs'][graph]] for graph in ['encoder','decoder']}
    model_files = {model_spec['models'][name]['path']: {k:model_spec['models'][name][k] for k in ['bytes','sha256']} for name in names}
    for name in names:
        path = ROOT / 'models/parakeet-tdt-0.6b-v3' / name
        assert pin(path) == {k:model_spec['models'][name][k] for k in ['bytes','sha256']}
        originals[path.relative_to(ROOT).as_posix()] = pin(path)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py']: copy(TOOLS / name, bundle / 'tools' / name)
    identities = {}
    for role, folder in [('selected', PARENT), ('candidate', BUILD)]:
        for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll', 'Google.Protobuf.dll']:
            copy(folder / 'collected/runtime' / name, bundle / 'products' / role / name)
        identities[role] = {name: pin(bundle / 'products' / role / name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
        assert identities[role] == read(folder / 'analysis.json')['built']
    for name in ['closed.json', 'analysis.json', 'payload.json']: copy(BUILD / name, bundle / 'evidence' / name)
    copy(BUILD / 'collected/collection.json', bundle / 'evidence/collection.json')
    copy(SOURCE / 'prepared.json', bundle / 'evidence/source-prepared.json')
    copy(TOOLS / 'README.md', bundle / 'prospective-residency.md')
    for name in ['closed.json','payload.json']: copy(CONTRACTS / name, bundle / 'evidence/contracts' / name)
    copy(CONTRACTS / 'collected/collection.json', bundle / 'evidence/contracts/collection.json')
    save(bundle / 'stage.json', dict(passed=True, identities=identities, models=models, model_files=model_files, budgets=dict(encoder=256*1024**2,decoder=64*1024**2),
         source_prepared=pin(SOURCE / 'prepared.json'), parent_build=pin(BUILD / 'closed.json'),
         files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals, stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz'), model_loads=8)))


if __name__ == '__main__': prepare()
