"""Freeze the block candidate only after its M54 parent is a qualified root release."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from checks import HELPER

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-scalar-where-build-amd-v3-20260924'
SOURCE = ROOT / 'artifacts/parakeet-scalar-where-source-v3-20260924'
QUALIFIED = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
RELEASE = ROOT / 'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'


def retained_scope():
    previous = TOOLS.parent / 'scalar-where-build-amd-v2'
    for name in ['protocol.py', 'remote.py', 'run.py', 'audit.py', 'il_body.py', 'remote_prepare.py']:
        expected = (previous / name).read_text().replace('scalar-where-build-amd-v2-20260924', 'scalar-where-build-amd-v3-20260924').replace('scalar-where-build-v2-20260924', 'scalar-where-build-v3-20260924')
        assert (TOOLS / name).read_text() == expected, name
    previous = (ROOT / 'tests/benchmarks/warmed-release-amd-v2/checks.py').read_text()
    start = previous.index('def normalized_body('); end = previous.index('\n\ndef consumer_inventory', start)
    actual = (TOOLS / 'il_body.py').read_text()
    assert actual[actual.index('def normalized_body('):].strip() == previous[start:end].strip()


def previous_closed():
    retained_scope()
    assert pin(SOURCE / 'prepared.json')['sha256'] == '6971271ecaa50346aa43b5040dab55cf6c9a812282aedbe34b5f8d117a700c89'
    source = read(SOURCE / 'prepared.json')
    assert source['passed'] and not source['built'] and not source['root_product_changed']
    assert len(source['before']) == 422 and len(source['source']) == 423
    assert source['changed'] == ['src/Lokad.Onnx/TensorOps.Elementwise.cs', 'src/Lokad.Onnx/Zzz.UniformScalarWhere.cs']
    for name, wanted in source['source'].items():
        assert pin(SOURCE / 'source' / name) == wanted, name
    for name, key in [('candidate.patch', 'patch'), ('prospective-plan.md', 'plan')]:
        assert pin(SOURCE / name) == source[key]
    assert pin(ROOT / 'tests/parakeet/scalar-where-source-v3/prepare.py') == source['generator']
    assert pin(ROOT / 'tests/parakeet/scalar-where-source-v3/UniformScalarWhere.cs') == source['helper']
    assert pin(QUALIFIED / 'closed.json')['sha256'] == 'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'
    for folder in [QUALIFIED, RELEASE]:
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items():
            assert pin(folder / name) == wanted, name
    release = read(RELEASE / 'analysis.json')
    assert release['root_source_verified'] and release['measured'] == read(QUALIFIED / 'analysis.json')['built']
    assert release['inventory'] == dict(passed=True, core_methods=3189, data_methods=697,
                                        public_surface_equal=True, implementation_flags_equal=True)
    applied = read(RELEASE / 'bundle/evidence/root-applied.json')
    assert applied['passed'] and applied['source_files'] == source['before']
    for name, wanted in source['before'].items():
        assert pin(ROOT / name) == wanted, name
    assert not (ROOT / source['changed'][1]).exists()


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    source = read(SOURCE / 'prepared.json')
    for name in source['source']:
        copy(SOURCE / 'source' / name, bundle / 'source' / name)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'il_body.py']:
        copy(TOOLS / name, bundle / 'tools' / name)
    for name in ['Bridge.dll', 'Bridge.deps.json', 'Bridge.runtimeconfig.json']:
        copy(QUALIFIED / 'collected/bridge' / name, bundle / 'bridge' / name)
    for name in ['closed.json', 'analysis.json', 'payload.json']:
        copy(QUALIFIED / name, bundle / 'evidence' / name)
        copy(RELEASE / name, bundle / 'evidence/release' / name)
    copy(QUALIFIED / 'collected/collection.json', bundle / 'evidence/collection.json')
    copy(RELEASE / 'collected/collection.json', bundle / 'evidence/release/collection.json')
    copy(RELEASE / 'bundle/evidence/root-applied.json', bundle / 'evidence/release/root-applied.json')
    for name in ['prepared.json', 'candidate.patch', 'prospective-plan.md']:
        copy(SOURCE / name, bundle / 'evidence' / ('source-' + name))
    copy(TOOLS / 'README.md', bundle / 'prospective-build.md')
    path = QUALIFIED / 'collected/inventory/instructions.json'
    row = read(path)['observations'][0]; methods = row['normalized_methods'] | row['candidate_methods']
    save(bundle / 'evidence/prior-composition.json', dict(passed=True, inventory=pin(path),
         closure=pin(QUALIFIED / 'closed.json'), helper_key=HELPER, helper_body=methods[HELPER]))
    originals[path.relative_to(ROOT).as_posix()] = pin(path)
    current = QUALIFIED / 'collected/runtime'
    stage = dict(passed=True, source_prepared=pin(SOURCE / 'prepared.json'),
                 parent_release=pin(RELEASE / 'closed.json'),
                 measured={name: pin(current / name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']},
                 measured_files={p.name: pin(p) for p in current.iterdir() if p.is_file()},
                 files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert stage['measured'] == read(QUALIFIED / 'analysis.json')['built']
    assert sum(name.startswith('source/') for name in stage['files']) == 423
    save(bundle / 'stage.json', stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    for p in [*(TOOLS.parent / 'scalar-where-build-amd-v2' / name for name in ['protocol.py', 'remote.py', 'run.py', 'audit.py', 'il_body.py', 'remote_prepare.py']),
              ROOT / 'tests/benchmarks/warmed-release-amd-v2/checks.py']:
        originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals, stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), stage=pin(bundle / 'stage.json'), source_files=423)))


if __name__ == '__main__':
    prepare()
