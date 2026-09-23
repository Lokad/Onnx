"""Freeze the selected source, four-job build and exact isolation proof."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from checks import one, GENERAL

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-isolated-short-kernels-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-isolated-short-kernels-source-20260923'
QUALIFIED = ROOT / 'artifacts/pyannote-winograd-product-root-amd-20260923'
INSPECTOR = ROOT / 'artifacts/parakeet-first-use-kernels-build-amd-20260923'


def previous_closed():
    assert pin(SOURCE / 'prepared.json')['sha256'] == 'f6635fac6f21ec4832ea105fc7d4e64888db8716b1f4c345776652cc30989104'
    source = read(SOURCE / 'prepared.json')
    assert source['passed'] and not source['root_product_changed'] and not source['built']
    assert len(source['source']) == 421 and len(source['before']) == 420
    assert source['changed'] == ['src/Lokad.Onnx/TensorOps.MatMul.cs'] and source['added'] == ['src/Lokad.Onnx/Zzz.IsolatedShortMatMul.cs']
    for name, wanted in source['source'].items(): assert pin(SOURCE / 'source' / name) == wanted, name
    for name, wanted in source['before'].items(): assert pin(ROOT / name) == wanted, name
    for name, key in [('candidate.patch', 'patch'), ('prospective-plan.md', 'plan'), ('census.json', 'census')]:
        assert pin(SOURCE / name) == source[key]
    assert source['generator'] == pin(ROOT / 'tests/parakeet/isolated-short-kernels-source/prepare.py')
    assert pin(QUALIFIED / 'closed.json') == source['qualified_parent']
    assert source['qualified_parent']['sha256'] == '62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'
    for folder in [QUALIFIED, INSPECTOR]:
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    assert pin(INSPECTOR / 'closed.json')['sha256'] == '2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243'


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    source = read(SOURCE / 'prepared.json')
    for name in source['source']: copy(SOURCE / 'source' / name, bundle / 'source' / name)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py']: copy(TOOLS / name, bundle / 'tools' / name)
    for name in ['Bridge.dll', 'Bridge.deps.json', 'Bridge.runtimeconfig.json']: copy(INSPECTOR / 'collected/bridge' / name, bundle / 'bridge' / name)
    copy(INSPECTOR / 'closed.json', bundle / 'evidence/inspector-closed.json')
    for name in ['closed.json', 'analysis.json', 'payload.json']: copy(QUALIFIED / name, bundle / 'evidence' / name)
    copy(QUALIFIED / 'collected/collection.json', bundle / 'evidence/collection.json')
    for name in ['prepared.json', 'candidate.patch', 'prospective-plan.md']:
        copy(SOURCE / name, bundle / 'evidence' / ('source-' + name))
    copy(TOOLS / 'README.md', bundle / 'prospective-build.md')
    path = INSPECTOR / 'collected/inventory/instructions.json'
    row = read(path)['observations'][0]
    wrapper = one(row['candidate_methods'], 'Lokad.Onnx.Tensor`1[T]::' + GENERAL + '::')
    short = wrapper.replace(GENERAL, 'RunShortWidePackedRows')
    general = wrapper.replace(GENERAL, 'RunGeneralFloatMatMulKernel')
    from checks import operand
    save(bundle / 'evidence/prior-composition.json', dict(passed=True, inventory=pin(path), closure=pin(INSPECTOR / 'closed.json'),
        wrapper_body=row['candidate_methods'][wrapper], short_body=row['candidate_methods'][short],
        original_general_body=row['normalized_methods'][wrapper], old_short_operand=operand(short), old_general_operand=operand(general)))
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
