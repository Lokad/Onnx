"""Freeze exact contract sources and qualified DLLs without rebuilding products."""
import ast
import json
from pathlib import Path
import re
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-inclusive-packing-contracts-amd-20260924'
BUILD = ROOT / 'artifacts/parakeet-inclusive-packing-build-amd-20260924'
SOURCE = ROOT / 'artifacts/parakeet-inclusive-packing-source-20260924'
PARENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
TESTS = ['FoldBudgetTests.cs', 'PackedBoundaryTests.cs', 'PackedWeightsTests.cs', 'PackedWeightBudgetTests.cs']


def previous_closed():
    for folder, digest in [(BUILD, '50f2a3a8a2ebbe20d41315bc0be4242e24fb7f3eb74e78533091a750791a4078'),
                           (PARENT, 'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58')]:
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
    expected = {'Lokad.Onnx.Backend.Tests.IdentityTests.ConsumedProductsAndInstructionModeMatch': 1}
    for name, count in zip(TESTS, [17, 12, 16, 16], strict=True):
        path = SOURCE / 'source/tests/Lokad.Onnx.Backend.Tests' / name
        copy(path, bundle / 'source' / name)
        found = re.findall(r'((?:\s*\[(?:Fact|Theory|InlineData\([^\n]*\))\]\s*)+)public void (\w+)', path.read_text())
        methods = {'Lokad.Onnx.Backend.Tests.' + name[:-3] + '.' + method: attributes.count('[InlineData(') or 1 for attributes, method in found}
        assert sum(methods.values()) == count; expected.update(methods)
    assert sum(expected.values()) == 62
    for name in ['IdentityTests.cs', 'PackingContracts.csproj']: copy(TOOLS / name, bundle / 'source' / name)
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
    copy(TOOLS / 'README.md', bundle / 'prospective-contracts.md')
    save(bundle / 'stage.json', dict(passed=True, identities=identities, expected_cases=expected,
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
    print(json.dumps(dict(stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz'), tests=62)))


if __name__ == '__main__': prepare()
