"""Freeze the original shared/e5 consumer and every unchanged native fixture."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save, RETAINED
from scope import census

ROOT = Path(__file__).resolve().parents[3]; TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-winograd-product-shared-amd-v2-20260923'
MODELS = ROOT/'artifacts/pyannote-winograd-product-models-amd-v2-20260923'
FAILED = ROOT/'artifacts/pyannote-winograd-product-shared-amd-20260923'
OLD = ROOT/'artifacts/e5-profiler-shared-v2-20260921'
REFERENCE = ROOT/'artifacts/shared-regression-20260918/reference'
E5 = ROOT/'artifacts/e5-randomized-processes-20260921/payload/inputs'
CASES = ['e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok']
REPLAY = 'a50d3e965cf480844559b1f5856e3de318b5c8a269742a9e6504afb9cee6c8c0'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('shared_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)


def previous_closed():
    assert pin(FAILED/'closed.json')['sha256'] == '053b003ef278896884f8214a5f11316b63ddf7a922c80688302f5987f9bec902'
    failure=read(FAILED/'closed.json');assert not failure['passed'] and failure['retained_failure']
    for name,wanted in failure['files'].items():assert pin(FAILED/name)==wanted,name
    assert pin(MODELS/'closed.json')['sha256'] == '82021de9672fb54cc684cc03c35e3d8c84ffb7b180dfcba7fa2c8a03f3296865'
    for name, wanted in read(MODELS/'closed.json')['files'].items(): assert pin(MODELS/name) == wanted, name
    assert pin(OLD/'closed.json')['sha256'] == '77f2ab0b4761b09be7225b69667e071a05826fb0c053a67b1e6d2a93a99ea533'
    assert read(OLD/'closed.json')['qualified']


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir()
    bundle = BASE/'bundle'; bundle.mkdir(); originals = {}; provenance = {}; external = {}
    prior = read(OLD/'closed.json')['files']
    def copy(source, target, historical=False):
        wanted = pin(source); name = source.relative_to(ROOT).as_posix()
        if historical: assert prior[name] == wanted, name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[name] = wanted; provenance[target.relative_to(bundle).as_posix()] = dict(source=name, **wanted)
    def model(source, wanted):
        name = source.relative_to(ROOT).as_posix()
        assert pin(source) == prior[name] == wanted, name
        originals[name] = wanted; external['/home/vermorel/Onnx/'+name] = wanted
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py']: copy(TOOLS/name, bundle/'tools'/name)
    for name in ['Replay.dll', 'Replay.deps.json', 'Replay.runtimeconfig.json', 'Google.Protobuf.dll']:
        copy(OLD/'runtimes/baseline'/name, bundle/'consumer'/name, True)
    assert pin(bundle/'consumer/Replay.dll')['sha256'] == REPLAY
    copy(REFERENCE/'manifest.json', bundle/'reference/manifest.json', True)
    for entry in read(REFERENCE/'manifest.json')['models']:
        for asset in entry['assets']: model(ROOT/asset['file'], {k: asset[k] for k in ['bytes', 'sha256']})
        for scenario in entry['scenarios']:
            for step in scenario['steps']:
                for item in step['inputs']+step['outputs']:
                    source = REFERENCE/item['file']; assert pin(source)['sha256'] == item['sha256']
                    if not (bundle/'reference'/item['file']).exists(): copy(source, bundle/'reference'/item['file'], True)
    for name in CASES:
        source = E5/(name+'.json'); copy(source, bundle/'e5'/source.name, True); fixture = read(source)
        array = E5/fixture['reference_file']; assert pin(array)['sha256'] == fixture['reference_sha256']
        copy(array, bundle/'e5'/array.name, True)
        source = ROOT/'models/multilingual-e5-small/model.onnx'; wanted = pin(source)
        assert wanted['sha256'] == fixture['model_sha256']; model(source, wanted)
    for mode in ['shared','e5']:
        copy(OLD/'outputs'/(mode+'-0-baseline')/'result.json', bundle/'evidence'/(mode+'-historical.json'), True)
    for name in ['closed.json','analysis.json']: copy(MODELS/name, bundle/'evidence'/('models-'+name))
    copy(ROOT/'tests/parakeet/reduction-shared/qualify_v2.py', bundle/'evidence/original-auditor.py')
    copy(ROOT/'tests/e5/fingerprint-product/Program.cs', bundle/'evidence/original-consumer.cs')
    copy(ROOT/'PLAN.md', bundle/'prospective-plan.md'); originals.pop('PLAN.md')
    for name in ['closed.json','analysis.json']:
        copy(FAILED/name,bundle/'evidence'/('failed-'+name))
    for job in RETAINED:
        for source in (FAILED/'collected'/job/'output').iterdir():copy(source,bundle/job/'output'/source.name)
    scope=census(ROOT,read(REFERENCE/'manifest.json'))
    save(bundle/'arithmetic-scope.json',scope)
    save(bundle/'provenance.json', provenance)
    stage = dict(passed=True, arithmetic_scope=scope, retained_failure=pin(FAILED/'closed.json'), identities=read(MODELS/'analysis.json')['identities'], consumer=pin(bundle/'consumer/Replay.dll'),
        model_assets=external, runtime='10.0.8', files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    files = dict(originals)
    for p in [*TOOLS.iterdir(), MONITOR, MODELS/'closed.json', OLD/'closed.json', FAILED/'closed.json']:
        if p.is_file(): files[p.relative_to(ROOT).as_posix()] = pin(p)
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=files, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'), model_assets=len(external))))


if __name__ == '__main__': prepare()
