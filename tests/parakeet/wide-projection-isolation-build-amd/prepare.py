"""Freeze the one-guard extension and compare actual normal-build DLLs to M50."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from checks import DISPATCH, extended_body

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-wide-projection-isolation-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-wide-projection-isolation-source-20260923'
QUALIFIED = ROOT / 'artifacts/parakeet-isolated-short-kernels-build-amd-v2-20260923'
DIAG = ROOT / 'artifacts/parakeet-isolated-runtime-diagnostic-amd-20260923'


def previous_closed():
    assert pin(SOURCE / 'prepared.json')['sha256'] == '2714b31148e581466fad1802f1d13560d860950a39481eee40eb59100b8550ee'
    source = read(SOURCE / 'prepared.json'); assert source['passed'] and not source['root_product_changed'] and not source['built']
    assert len(source['source']) == 421 and len(source['before']) == 420
    assert source['delta_from_parent'] == ['src/Lokad.Onnx/Zzz.IsolatedShortMatMul.cs']
    for n, v in source['source'].items(): assert pin(SOURCE / 'source' / n) == v, n
    for n, v in source['before'].items(): assert pin(ROOT / n) == v, n
    for n, key in [('candidate.patch', 'patch'), ('prospective-plan.md', 'plan'), ('census.json', 'census')]: assert pin(SOURCE / n) == source[key]
    assert source['generator'] == pin(ROOT / 'tests/parakeet/wide-projection-isolation-source/prepare.py')
    for folder, digest in [(QUALIFIED, '0adb72e2eae376df7d4f3dd6eb7f4a97b15dd57c5d4c3fa962f8a4cd3a9c45c6'),
                           (DIAG, 'ecfea393d0475617f32de3b87419cd6354f6d3b61edd4883adbdba877caa0f2c')]:
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
    keys = [k for k in row['candidate_methods'] if '::' + DISPATCH + '::' in k]; assert len(keys) == 1
    key = keys[0]; body = row['candidate_methods'][key]; extended_body(body)
    save(bundle / 'evidence/prior-composition.json', dict(passed=True, inventory=pin(path), closure=pin(QUALIFIED / 'closed.json'),
        dispatcher_key=key, dispatcher_body=body))
    originals[path.relative_to(ROOT).as_posix()] = pin(path)
    current = QUALIFIED / 'collected/runtime'
    stage = dict(passed=True, source_prepared=pin(SOURCE / 'prepared.json'),
        measured={n: pin(current / n) for n in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']},
        measured_files={p.name: pin(p) for p in current.iterdir() if p.is_file()},
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert stage['measured'] == read(QUALIFIED / 'analysis.json')['built']
    assert sum(n.startswith('source/') for n in stage['files']) == 421
    save(bundle / 'stage.json', stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals, stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), stage=pin(bundle / 'stage.json'), source_files=421)))


if __name__ == '__main__': prepare()
