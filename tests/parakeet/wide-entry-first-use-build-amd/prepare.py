"""Freeze the wide entry clone and compare actual normal-build DLLs to M52."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from checks import ENTRY, one

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-wide-entry-first-use-source-v2-20260923'
QUALIFIED = ROOT / 'artifacts/parakeet-wide-projection-isolation-build-amd-20260923'
DIAG = ROOT / 'artifacts/parakeet-wide-runtime-diagnostic-amd-20260923'


def previous_closed():
    assert pin(SOURCE / 'prepared.json')['sha256'] == '72c22bee93652ed8d6a759c2a984d965fa341697e869cf7a2461a908727483f6'
    source = read(SOURCE / 'prepared.json'); assert source['passed'] and not source['root_product_changed'] and not source['built']
    assert len(source['source']) == 422 and len(source['before']) == 420
    assert source['delta_from_parent'] == ['src/Lokad.Onnx/TensorOps.MatMul.cs', 'src/Lokad.Onnx/Zzz.IsolatedShortMatMul.cs', 'src/Lokad.Onnx/Zzz.WideProjectionEntry.cs']
    for n, v in source['source'].items(): assert pin(SOURCE / 'source' / n) == v, n
    for n, v in source['before'].items(): assert pin(ROOT / n) == v, n
    for n, key in [('candidate.patch', 'patch'), ('prospective-plan.md', 'plan'), ('census.json', 'census')]: assert pin(SOURCE / n) == source[key]
    assert source['generator'] == pin(ROOT / 'tests/parakeet/wide-entry-first-use-source-v2/prepare.py')
    for folder, digest in [(QUALIFIED, 'c67b21b2e3f9d3d09f822ab075e649e232c6c42d7e7503470711d21a1bf5604d'),
                           (DIAG, 'fee8b6d9a20917ca24a03300f35720d644b7841ba2ad5408f71392f11bf7642c')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for n, v in proof['files'].items(): assert pin(folder / n) == v, n
    assert source['diagnostic'] == pin(DIAG / 'closed.json')


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    source = read(SOURCE / 'prepared.json')
    for n in source['source']: copy(SOURCE / 'source' / n, bundle / 'source' / n)
    for n in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py']: copy(TOOLS / n, bundle / 'tools' / n)
    for n in ['Bridge.dll', 'Bridge.deps.json', 'Bridge.runtimeconfig.json']: copy(QUALIFIED / 'collected/bridge' / n, bundle / 'bridge' / n)
    for n in ['closed.json', 'analysis.json', 'payload.json']: copy(QUALIFIED / n, bundle / 'evidence' / n)
    copy(QUALIFIED / 'collected/collection.json', bundle / 'evidence/collection.json')
    copy(DIAG / 'collected/collection.json', bundle / 'evidence/diagnostic-collection.json')
    for n in ['prepared.json', 'candidate.patch', 'prospective-plan.md']: copy(SOURCE / n, bundle / 'evidence' / ('source-' + n))
    copy(TOOLS / 'README.md', bundle / 'prospective-build.md')
    path = QUALIFIED / 'collected/inventory/instructions.json'; row = read(path)['observations'][0]
    methods = row['normalized_methods'] | row['candidate_methods']
    key = one(methods, 'Lokad.Onnx.Tensor`1[T]::' + ENTRY + '::')
    body = methods[key]
    save(bundle / 'evidence/prior-composition.json', dict(passed=True, inventory=pin(path), closure=pin(QUALIFIED / 'closed.json'),
        entry_key=key, entry_body=body))
    originals[path.relative_to(ROOT).as_posix()] = pin(path)
    current = QUALIFIED / 'collected/runtime'
    stage = dict(passed=True, source_prepared=pin(SOURCE / 'prepared.json'),
        measured={n: pin(current / n) for n in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']},
        measured_files={p.name: pin(p) for p in current.iterdir() if p.is_file()},
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert stage['measured'] == read(QUALIFIED / 'analysis.json')['built']
    assert sum(n.startswith('source/') for n in stage['files']) == 422
    save(bundle / 'stage.json', stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals, stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), stage=pin(bundle / 'stage.json'), source_files=422)))


if __name__ == '__main__': prepare()
