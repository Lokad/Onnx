"""Freeze the one observed-layout candidate against the qualified current release."""
import ast
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from checks import PROVIDER, ADDED

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-observed-dense-where-build-amd-20260924'
SOURCE = ROOT/'artifacts/parakeet-observed-dense-where-source-20260924'
CURRENT = ROOT/'artifacts/parakeet-validated-composition-build-amd-v2-20260924'
RELEASE = ROOT/'artifacts/parakeet-validated-composition-root-amd-20260924'
RETAINED = ROOT/'artifacts/parakeet-dense-scalar-where-build-amd-20260924'


def verify_closed(folder, digest):
    assert pin(folder/'closed.json')['sha256'] == digest
    proof = read(folder/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name


def previous_closed():
    assert pin(SOURCE/'prepared.json')['sha256'] == '41f2477c7060c3f543471bebe991c77ea3b853d724708fec098a2cbb505cb0a2'
    source = read(SOURCE/'prepared.json')
    assert source['passed'] and len(source['source']) == 426 and len(source['before']) == 425
    assert source['changed'] == ['src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs', 'src/Lokad.Onnx/Zzz.DenseScalarWhere.cs']
    for name, wanted in source['source'].items(): assert pin(SOURCE/'source'/name) == wanted, name
    for name, wanted in source['before'].items(): assert pin(ROOT/name) == wanted, name
    for key, name in [('patch','candidate.patch'), ('plan','prospective-plan.md')]: assert pin(SOURCE/name) == source[key]
    assert pin(TOOLS.parent/'observed-dense-where-source/prepare.py') == source['generator']
    verify_closed(RELEASE, 'c7a1d2e11566e6eeeb965de6c9cedbf47df479fd51f194c797af412446281609')
    assert pin(CURRENT/'closed.json')['sha256'] == 'bb578d7ad6dfeaf989726476d6d7864438e602a41c7727a99ee05cb8c9eb71da'
    proof = read(CURRENT/'closed.json'); assert proof['passed']
    for key, name in [('analysis','analysis.json'), ('build_review','build-review.json'),
                      ('collection','capture-collected/capture-collection.json'), ('transfer','capture-transfer.json')]:
        assert proof[key] == pin(CURRENT/name), name
    for kind in ['build','capture']:
        for name, wanted in read(CURRENT/f'{kind}-collected/{kind}-collection.json')['files'].items():
            assert pin(CURRENT/f'{kind}-collected'/name) == wanted, name
    verify_closed(RETAINED, '1df6868777c87e124e59d24dcc313b5db5cd7e6d46c018a22336dd3da5d5a6d8')
    assert read(RELEASE/'analysis.json')['measured'] == source['measured']


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    source = read(SOURCE/'prepared.json')
    for name in source['source']: copy(SOURCE/'source'/name, bundle/'source'/name)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','il_body.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']:
        copy(RETAINED/'collected/bridge'/name, bundle/'bridge'/name)
    for name in ['prepared.json','candidate.patch','prospective-plan.md']:
        copy(SOURCE/name, bundle/'evidence'/('source-'+name))
    copy(TOOLS/'README.md', bundle/'prospective-build.md')
    current_inventory = RELEASE/'collected/inventory/instructions.json'
    old_inventory = RETAINED/'collected/inventory/instructions.json'
    originals[current_inventory.relative_to(ROOT).as_posix()] = pin(current_inventory)
    originals[old_inventory.relative_to(ROOT).as_posix()] = pin(old_inventory)
    rows = read(current_inventory)['observations']
    assert len(rows) == 2 and all(not r['differences'] and not r['added'] and not r['removed'] and r['compiler_rename'] is None for r in rows)
    old = read(old_inventory)['observations'][0]
    assert set(old['candidate_methods']) == {PROVIDER, *ADDED}
    save(bundle/'evidence/prior-composition.json', dict(passed=True, inventory=pin(current_inventory),
        retained_inventory=pin(old_inventory), current_methods={r['assembly']:r['normalized_methods'] for r in rows},
        current_flags={r['assembly']:r['method_flags_before'] for r in rows},
        retained_methods=old['candidate_methods'], retained_flags={k:old['method_flags_after'][k] for k in old['candidate_methods']}))
    for kind in ['build','capture']:
        copy(CURRENT/f'{kind}-collected/{kind}-collection.json', bundle/'evidence'/f'parent-{kind}-collection.json')
        copy(CURRENT/f'{kind}-collected/{kind}-state.json', bundle/'evidence'/f'parent-{kind}-state.json')
    copy(CURRENT/'bundle/spec.json', bundle/'evidence/parent-spec.json')
    previous = read(RETAINED/'payload.json')
    external = {k:v for k,v in previous['external'].items() if k.startswith('/home/vermorel/.dotnet/') or k.startswith(previous['feed']+'/')}
    assert external and any(k.endswith('.nupkg') for k in external)
    current = CURRENT/'build-collected/source/runtime-observed'
    assert {n:pin(current/n) for n in source['measured']} == source['measured']
    stage = dict(passed=True, source_prepared=pin(SOURCE/'prepared.json'), parent_release=pin(RELEASE/'closed.json'),
        measured=source['measured'], measured_files={p.name:pin(p) for p in current.iterdir() if p.is_file()},
        parent_remote='/dev/shm/lokad-parakeet-validated-composition-build-v2-20260924',
        feed=previous['feed'], external=external, interpreter=previous['interpreter'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert sum(n.startswith('source/') for n in stage['files']) == 426
    save(bundle/'stage.json', stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(__import__('json').dumps(dict(archive=pin(BASE/'payload.tar.gz'), source_files=426, measured=source['measured'])))


if __name__ == '__main__': prepare()
