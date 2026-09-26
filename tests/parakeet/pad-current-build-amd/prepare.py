"""Freeze the existing dispatcher over the qualified 435-file current product."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-pad-current-build-amd-20260926'
SOURCE = ROOT / 'artifacts/parakeet-pad-current-source-20260926'
QUALIFIED = ROOT / 'artifacts/parakeet-owned-batch-isolation-root-policy-amd-20260925'
INSPECTOR = ROOT / 'artifacts/parakeet-first-use-kernels-build-amd-20260923'


def previous_closed():
    source = read(SOURCE / 'prepared.json')
    assert source['passed'] and not source['built'] and not source['root_product_changed'] and not source['release_admitted']
    assert len(source['source']) == 437 and len(source['before']) == 435
    assert source['changed'] == ['src/Lokad.Onnx/CPUExecutionProvider.Shape.cs',
        'src/Lokad.Onnx/Zzz.LastAxisPadDispatch.cs', 'tests/Lokad.Onnx.Backend.Tests/LastAxisPadTests.cs']
    for name, wanted in source['source'].items(): assert pin(SOURCE / 'source' / name) == wanted, name
    for name, wanted in source['before'].items(): assert pin(ROOT / name) == wanted, name
    for name, key in [('candidate.patch', 'patch'), ('prospective-plan.md', 'plan')]: assert pin(SOURCE / name) == source[key]
    for name, wanted in source['tools'].items(): assert pin(ROOT / 'tests/parakeet/pad-current-source' / name) == wanted
    assert pin(QUALIFIED / 'closed.json') == source['qualified_parent']
    assert source['qualified_parent']['sha256'] == 'c77ef606c76508144c5656bdbe48aeac8948ce28396c8bee645587d31609c475'
    proof = read(QUALIFIED / 'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(QUALIFIED / name) == wanted, name
    assert read(QUALIFIED / 'analysis.json')['built'] == source['parent_product']
    assert pin(INSPECTOR / 'closed.json')['sha256'] == '2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243'
    old = read(INSPECTOR / 'closed.json'); assert old['passed']
    for name in ('Bridge.dll', 'Bridge.deps.json', 'Bridge.runtimeconfig.json'):
        assert pin(INSPECTOR / 'collected/bridge' / name) == old['files']['collected/bridge/' + name]
    reference = read(TOOLS / 'helper-reference.json')
    original = ROOT / 'artifacts/parakeet-pad-dispatch-build-amd-20260923'
    assert pin(original / 'closed.json') == reference['build']
    assert pin(original / 'collected/inventory/instructions.json') == reference['inventory']


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    source = read(SOURCE / 'prepared.json')
    for name in source['source']: copy(SOURCE / 'source' / name, bundle / 'source' / name)
    for name in ('protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'helper-reference.json'):
        copy(TOOLS / name, bundle / 'tools' / name)
    for name in ('Bridge.dll', 'Bridge.deps.json', 'Bridge.runtimeconfig.json'):
        copy(INSPECTOR / 'collected/bridge' / name, bundle / 'bridge' / name)
    copy(INSPECTOR / 'closed.json', bundle / 'evidence/inspector-closed.json')
    for name in ('closed.json', 'analysis.json', 'payload.json'):
        copy(QUALIFIED / name, bundle / 'evidence' / name)
    copy(QUALIFIED / 'collected/collection.json', bundle / 'evidence/collection.json')
    for name in ('prepared.json', 'candidate.patch', 'prospective-plan.md'):
        copy(SOURCE / name, bundle / 'evidence' / ('source-' + name))
    copy(TOOLS / 'README.md', bundle / 'prospective-build.md')
    runtime = QUALIFIED / 'collected/runtime'
    stage = dict(passed=True, source_prepared=pin(SOURCE / 'prepared.json'),
        measured=source['parent_product'], measured_files={p.name: pin(p) for p in runtime.iterdir() if p.is_file()},
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert all(stage['measured_files'][name] == value for name, value in stage['measured'].items())
    assert sum(n.startswith('source/') for n in stage['files']) == 437
    save(bundle / 'stage.json', stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals,
        stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), source_files=437)))


if __name__ == '__main__': prepare()
