"""Rebind the unchanged graph comparison to the one qualified relocation."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'packed-final-row-graphs-amd'
sys.path.insert(1, str(PARENT))
from protocol import pin, read, save, JOBS, LIMITS
from consumer_scope import verify_scope

BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-graphs-amd-20260925'
OLD = ROOT/'artifacts/parakeet-packed-final-row-graphs-amd-20260925'
BUILD = ROOT/'artifacts/parakeet-owned-batch-isolation-build-amd-20260925'
SOURCE = ROOT/'artifacts/parakeet-owned-batch-isolation-source-20260925'
REMOTE_OLD = '/dev/shm/lokad-parakeet-packed-final-row-graphs-20260925'
REMOTE_BUILD = '/dev/shm/lokad-parakeet-owned-batch-isolation-build-20260925'
WORKERS = ['protocol.py','remote.py','checks.py','checks_e5.py','native.py','native-e5.py']


def previous_closed():
    verify_scope()
    for folder, digest in [(OLD, '0b82805aa6de28287c9bc9242416b3abd5c58b76fdff57a0416208f29f3707f2'),
                           (BUILD, '5dd53e90d4924a36bb6f43cc80fa939e9236492949542dcf659dc056d4ea3860')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    old = read(OLD/'analysis.json')
    assert [r['key'] for r in old['performance'] if not r['regression_passed']] == ['e5-8tok']
    review = read(BUILD/'build-review.json')
    assert review['release_dispatcher_restored'] and not review['release_admitted']
    assert review['product']['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert review['source'] == pin(SOURCE/'prepared.json')
    source = read(SOURCE/'prepared.json')
    for name, wanted in source['source'].items(): assert pin(SOURCE/'source'/name) == wanted, name
    assert source['source_reversible'] and source['shared_dispatcher_matches_release']
    assert [s['passed'] for s in read(BUILD/'analysis.json')['suites']] == [64,25]
    return review


def prepare():
    review = previous_closed(); assert not BASE.exists()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = verify_scope()
    def copy(source, name):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target); originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in WORKERS: copy(PARENT/name, 'tools/'+name)
    copy(TOOLS/'remote_prepare.py', 'tools/remote_prepare.py')
    copy(TOOLS/'README.md', 'README.md')
    copy(ROOT/'.agent/e5-owned-batch-isolation-20260925.md', 'prospective-plan.md')
    # The plan is a snapshot, not a permanent mutable-file dependency.
    originals.pop('.agent/e5-owned-batch-isolation-20260925.md')
    old = read(OLD/'payload.json'); links = {}
    receipt = read(OLD/'collected/collection.json')
    for name, wanted in old['files'].items():
        if name.startswith(('reference/', 'runtimes/', 'runtimes-e5/', 'source/', 'evidence/')) or name in ['cases.json', 'built.json']:
            assert pin(OLD/'collected'/name) == wanted == receipt['files'][name], name
            links[name] = dict(source=REMOTE_OLD+'/'+name, identity=wanted)
    for prefix in ['runtimes', 'runtimes-e5']:
        name = prefix+'/candidate/Lokad.Onnx.dll'
        links[name] = dict(source=REMOTE_BUILD+'/runtime/Lokad.Onnx.dll', identity=review['product']['Lokad.Onnx.dll'])
    for name in ['closed.json', 'analysis.json', 'payload.json']:
        copy(OLD/name, 'evidence/parent-graphs/'+name)
    copy(OLD/'collected/collection.json', 'evidence/parent-graphs/collection.json')
    for name in ['closed.json', 'analysis.json', 'build-review.json']:
        copy(BUILD/name, 'evidence/isolation-build/'+name)
    copy(BUILD/'capture-collected/capture-collection.json', 'evidence/isolation-build/collection.json')
    copy(SOURCE/'prepared.json', 'evidence/isolation-source.json')
    products = dict(current=old['products']['current'], candidate={'Lokad.Onnx.dll':review['product']['Lokad.Onnx.dll']})
    stage = dict(passed=True, links=links, products=products, previous_owner=receipt['identities'][0],
                 consumer=old['consumer'], e5_consumer=old['e5_consumer'], previous_consumer=old['previous_consumer'],
                 external=old['external'], interpreter=old['interpreter'], python_paths=old['python_paths'],
                 files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert all(links[p+'/candidate/Lokad.Onnx.dll']['identity'] == products['candidate']['Lokad.Onnx.dll'] for p in ['runtimes','runtimes-e5'])
    save(bundle/'stage.json', stage)
    for folder in [TOOLS, PARENT]:
        for path in folder.iterdir():
            if path.is_file():
                if path.suffix == '.py': ast.parse(path.read_text(), str(path))
                originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'), jobs=len(JOBS), limits=LIMITS)))


if __name__ == '__main__':
    prepare()
