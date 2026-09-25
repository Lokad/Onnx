"""Bind exact products to complete Pyannote fixtures and existing consumers."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from consumer_scope import verify_scope

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-pyannote-amd-20260925'
CURRENT = ROOT/'artifacts/parakeet-observed-dense-where-pyannote-amd-20260924'
MODEL = CURRENT/'bundle'
APP_PAYLOAD = CURRENT/'collected'
PRODUCT = ROOT/'artifacts/parakeet-owned-batch-isolation-models-amd-20260925'
APP = ROOT/'artifacts/parakeet-owned-batch-isolation-release-app-amd-20260925'
SHARED = ROOT/'artifacts/parakeet-owned-batch-isolation-shared-amd-20260925'
REUSE = ROOT/'artifacts/parakeet-packed-final-row-pyannote-amd-20260925'
PRIOR = dict(current=CURRENT, product=PRODUCT, app=APP, shared=SHARED, reuse=REUSE)
OLD_DATA = 'a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
NEW_DATA = '01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
module = importlib.util.spec_from_file_location('models_monitor', MONITOR)
monitor = importlib.util.module_from_spec(module)
module.loader.exec_module(monitor)


def previous_closed():
    assert read(APP/'closed.json')['admitted'] and read(SHARED/'closed.json')['passed']
    verify_scope()
    digests = dict(current='e36fe9c83405608659ebdc4d673c185c8805a5414a6b01d401904209d59417dd',
        product='1997eeb782df89975f4893820bc0fc246dec60d1839b2c63a86ad2bba1aecd9d',
        reuse='8471ed0c4f9cdca7ba32edf2ba7ef744bedae4c0b0b037af52f1c87642230595')
    for label, folder in PRIOR.items():
        proof = read(folder/'closed.json')
        assert proof['passed'] and proof['analysis'] == pin(folder/'analysis.json')
        if label in digests:
            assert pin(folder/'closed.json')['sha256'] == digests[label]
        for name, wanted in proof['files'].items():
            assert pin(folder/name) == wanted, name
    selected = read(CURRENT/'analysis.json')['identities']['candidate']
    candidate = read(PRODUCT/'analysis.json')['identities']['candidate']
    assert selected['Lokad.Onnx.dll']['sha256'] == 'f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert selected['Lokad.Onnx.Data.dll']['sha256'] == OLD_DATA
    assert candidate['Lokad.Onnx.Data.dll']['sha256'] == NEW_DATA
    assert read(APP/'analysis.json')['identities'] == dict(current=selected, candidate=candidate)
    assert read(SHARED/'analysis.json')['identities'] == dict(selected=selected, candidate=candidate)
    assert read(REUSE/'analysis.json')['identities']['candidate']['Lokad.Onnx.Data.dll'] == candidate['Lokad.Onnx.Data.dll']


def prepare():
    assert not BASE.exists()
    previous_closed()
    BASE.mkdir()
    bundle = BASE/'bundle'
    bundle.mkdir()
    originals = verify_scope()

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py',
                 'candidate_protocol.py', 'qualify_outputs.py', 'reuse_checks.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    for role, folder in [('selected', CURRENT), ('candidate', REUSE)]:
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            name = 'GraphQualification.'+suffix
            copy(folder/'collected/built'/name, bundle/'consumers'/role/name)
    for label, folder in PRIOR.items():
        for name in ['closed.json', 'analysis.json', 'payload.json']:
            copy(folder/name, bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json', bundle/'evidence'/(label+'-collection.json'))
    for target, source in [('reuse-built.json', 'built.json'),
                           ('reuse-instructions.json', 'consumer-inventory/instructions.json'),
                           ('reuse-review.json', 'consumer-inventory/review.json')]:
        copy(REUSE/'collected'/source, bundle/'evidence'/target)
    copy(MODEL/'evidence/original-manifest.json', bundle/'evidence/original-manifest.json')
    copy(APP_PAYLOAD/'graph-reference.json', bundle/'graph-reference.json')
    copy(TOOLS/'README.md', bundle/'prospective-plan.md')
    stage = dict(passed=True,
        selected_product=read(CURRENT/'analysis.json')['identities']['candidate'],
        product=read(PRODUCT/'analysis.json')['identities']['candidate'],
        consumers={role: pin(bundle/'consumers'/role/'GraphQualification.dll') for role in ['selected', 'candidate']},
        old_data=OLD_DATA, new_data=NEW_DATA,
        files={path.relative_to(bundle).as_posix(): pin(path) for path in bundle.rglob('*') if path.is_file()})
    save(bundle/'stage.json', stage)
    for path in [*TOOLS.iterdir(), MONITOR]:
        if path.is_file():
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    for path in TOOLS.glob('*.py'):
        ast.parse(path.read_text(), str(path))
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'))))


if __name__ == '__main__':
    prepare()
