"""Freeze actual integrated root sources for a normal Linux build and package check."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
PRODUCT = ROOT/'artifacts/pyannote-lstm-input-product-amd-v2-20260922'
APP = ROOT/'artifacts/pyannote-lstm-input-app-amd-20260922'
APPLIED = ROOT/'artifacts/pyannote-lstm-input-root-integration-20260922/applied.json'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('product_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)
TEXT = {'.cs', '.csproj', '.props', '.targets', '.json', '.slnx', '.md', '.txt', '.config', '.py', '.proto'}


def canonical(path):
    data = path.read_bytes()
    return data.decode('utf-8-sig').replace('\r\n', '\n').encode('utf8') if path.suffix in TEXT else data


def previous_closed():
    for folder, digest in [(PRODUCT, '7d442059d85e5df31f9a616b4e5e5d5203a612affb41b59a0c1128e6e046529f'),
                           (APP, '73a4897a4db8e1bd729cb3c5486bcb11814d4ff669c9b9a472003572b08c64d0')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    assert read(APP/'analysis.json')['performance']['admitted']
    assert pin(APPLIED)['sha256'] == 'cc890faabf0f097d859583b649c1ee2bc573596243c95e96af44d1ef85652b63'
    for name, wanted in read(APPLIED)['source_files'].items(): assert pin(ROOT/name) == wanted, name


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    qualified = PRODUCT/'bundle'
    stage_before = read(qualified/'stage.json')
    for name, wanted in stage_before['files'].items():
        if name.startswith('source/'):
            relative = name.removeprefix('source/')
            assert pin(qualified/name) == wanted
            assert canonical(ROOT/relative) == canonical(qualified/name), name
            copy(ROOT/relative, bundle/name)
        elif name.startswith(('consumer/', 'bridge/', 'evidence/')) and name != 'evidence/tensor-source.tar':
            assert pin(qualified/name) == wanted
            copy(qualified/name, bundle/name)
    # Preserve the complete tensor-source member list, but archive actual root bytes.
    with tarfile.open(qualified/'evidence/tensor-source.tar') as before:
        names = [member.name for member in before.getmembers() if member.isfile()]
    with tarfile.open(bundle/'evidence/tensor-source.tar', 'w') as archive:
        for name in names: archive.add(ROOT/name, arcname=name, recursive=False)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    copy(APP/'closed.json', bundle/'evidence/application-closed.json')
    copy(APP/'analysis.json', bundle/'evidence/application-analysis.json')
    copy(APPLIED, bundle/'evidence/root-applied.json')
    copy(ROOT/'PLAN.md', bundle/'prospective-plan.md'); originals.pop('PLAN.md')
    stage = dict(passed=True, measured_core=read(APP/'analysis.json')['identities']['candidate']['Lokad.Onnx.dll'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_scope='Actual root working tree after admitted five-file M22 integration; exact qualified source after text newline normalization.',
        application=pin(APP/'closed.json'), applied=pin(APPLIED),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    files = dict(originals)
    for p in [*TOOLS.iterdir(), MONITOR, PRODUCT/'closed.json', APP/'closed.json', APPLIED]:
        if p.is_file(): files[p.relative_to(ROOT).as_posix()] = pin(p)
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=files, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'), source_files=sum(k.startswith('source/') for k in stage['files']))))


if __name__ == '__main__': prepare()
