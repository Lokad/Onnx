"""Freeze the existing Pyannote consumer and original model checks without builds."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from consumer_reuse import review
from consumer_scope import verify_scope

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-rational-sigmoid-pyannote-amd-20260927'
CURRENT = ROOT/'artifacts/parakeet-observed-dense-where-pyannote-amd-20260924'
MODEL = CURRENT/'bundle'; APP_PAYLOAD = CURRENT/'collected'
PREVIOUS = ROOT/'artifacts/parakeet-pad-current-pyannote-amd-20260926'
PRODUCT = ROOT/'artifacts/parakeet-rational-sigmoid-models-amd-20260927'
APP = ROOT/'artifacts/parakeet-rational-sigmoid-app-amd-20260927'
SHARED = ROOT/'artifacts/parakeet-rational-sigmoid-shared-amd-20260927'
PRIOR = dict(current=CURRENT, previous=PREVIOUS, product=PRODUCT, app=APP, shared=SHARED)
DIGESTS = dict(current='e36fe9c83405608659ebdc4d673c185c8805a5414a6b01d401904209d59417dd',
    previous='5e2b54bb685d9e954d3840524c486b55a22aadc5a574b6cc189bedf5bf487435',
    product='3d2e01b5aca3f66c434d5d1e8124c2b0d08327bc14dd37409c0e1884619769d8',
    app='629183e44963638c0285ea4daf70f8f9aab3b94d5e4c6164225d205aaaa443b4',
    shared='86ad6b3f5d11549dc968270e6ef5936787a266a84969de1932e750450ee6da65')
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
loader = importlib.util.spec_from_file_location('pyannote_monitor', MONITOR)
monitor = importlib.util.module_from_spec(loader); loader.loader.exec_module(monitor)


def validate(reports):
    assert all(r['passed'] for r in reports.values())
    models, app, shared = [reports[n] for n in ['product', 'app', 'shared']]
    products = models['identities']
    assert products['selected']['Lokad.Onnx.dll']['sha256'] == '8bb22038d0b4c09b56b2cdae06c49c165b8e646bc73ca28ad400f4ace0bfc659'
    assert products['candidate']['Lokad.Onnx.dll']['sha256'] == '946ddfb66492c48a0fc6078ecbe1957ac494ff5d9ff0be42259d70e66e3b1f24'
    assert shared['identities'] == products
    assert app['identities'] == dict(current=products['selected'], candidate=products['candidate'])
    assert app['performance']['admitted']
    for key, count in [('controls', 63), ('gates', 21)]:
        assert len(app['performance'][key]) == count and all(r['passed'] for r in app['performance'][key])
    assert reports['current']['reference_provenance_verified'] and shared['reference_provenance_verified']
    return products


def previous_closed():
    verify_scope(); reports = {}
    for label, folder in PRIOR.items():
        assert pin(folder/'closed.json')['sha256'] == DIGESTS[label]
        proof = read(folder/'closed.json'); assert proof['passed'] and proof['analysis'] == pin(folder/'analysis.json')
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
        reports[label] = read(folder/'analysis.json')
    assert read(APP/'closed.json')['admitted']
    products = validate(reports); reuse = review()
    assert reuse['identities'] == products
    for role in ['selected', 'candidate']:
        for name, wanted in products[role].items(): assert pin(PRODUCT/'collected/runtimes'/role/name) == wanted
    return products


def prepare():
    assert not BASE.exists(); products = previous_closed(); reuse = review()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = dict(verify_scope(), **reuse['inputs'])
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'candidate_protocol.py', 'qualify_outputs.py', 'identity_probes.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    for name in ['GraphQualification.dll', 'GraphQualification.deps.json', 'GraphQualification.runtimeconfig.json']:
        copy(PREVIOUS/'collected/built'/name, bundle/'consumer'/name)
    for label, folder in PRIOR.items():
        for name in ['closed.json', 'analysis.json', 'payload.json']:
            copy(folder/name, bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json', bundle/'evidence'/(label+'-collection.json'))
    copy(PRODUCT/'collected/evidence/compatibility.json', bundle/'evidence/product-compatibility.json')
    copy(PREVIOUS/'collected/built.json', bundle/'evidence/previous-built.json')
    save(bundle/'evidence/consumer-reuse.json', reuse)
    copy(MODEL/'evidence/original-manifest.json', bundle/'evidence/original-manifest.json')
    copy(APP_PAYLOAD/'graph-reference.json', bundle/'graph-reference.json')
    copy(TOOLS/'README.md', bundle/'prospective-pyannote.md')
    shutil.copy2(ROOT/'PLAN.md', bundle/'prospective-plan.md')
    stage = dict(passed=True, identities=products, consumer=reuse['consumer'],
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    for p in [*TOOLS.iterdir(), MONITOR]:
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'), consumer=reuse['consumer'])))


if __name__ == '__main__': prepare()
