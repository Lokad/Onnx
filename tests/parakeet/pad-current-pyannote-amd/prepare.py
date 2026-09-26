"""Freeze complete Pyannote qualification with one parameterized consumer."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save
from identity_scope import source
from consumer_scope import verify_scope

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-pad-current-pyannote-amd-20260926'
CURRENT=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-amd-20260924'
MODEL=CURRENT/'bundle';APP_PAYLOAD=CURRENT/'collected'
PRODUCT=ROOT/'artifacts/parakeet-pad-current-models-amd-20260926'
APP=ROOT/'artifacts/parakeet-pad-current-app-amd-20260926'
SHARED=ROOT/'artifacts/parakeet-pad-current-shared-amd-20260926'
PRIOR=dict(current=CURRENT,product=PRODUCT,app=APP,shared=SHARED)
DIGESTS=dict(current='e36fe9c83405608659ebdc4d673c185c8805a5414a6b01d401904209d59417dd',
    product='194eb9a48b64df22d19b51adfc3ebe3accc56004544620123c128c2a76f066bf',
    app='2b40b54d8e94de7326ceec5228ad1529c7513964211e8b5dfbb2c921cd798151',
    shared='d1cb211096aa6a02ea70601b93cb615186c5d3b00472dced1c53283a1a8d3e3c')
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
loader=importlib.util.spec_from_file_location('pyannote_monitor',MONITOR)
monitor=importlib.util.module_from_spec(loader);loader.loader.exec_module(monitor)


def validate(reports):
    current,models,app,shared=[reports[n] for n in ['current','product','app','shared']]
    assert all(r['passed'] for r in reports.values())
    products=models['identities']
    assert products['selected']['Lokad.Onnx.dll']['sha256']=='f3992f40d889a932cd0d15e0323564db801a308f4b24d1848731af30ba3c19f6'
    assert products['candidate']['Lokad.Onnx.dll']['sha256']=='a74acb17524f23be13e81ade871b2b2ffea2afde5bcdd12e339b75e0197edf10'
    assert shared['identities']==products
    assert app['identities']==dict(current=products['selected'],candidate=products['candidate'])
    assert app['performance']['admitted']
    assert len(app['performance']['controls'])==63 and all(r['passed'] for r in app['performance']['controls'])
    assert len(app['performance']['gates'])==21 and all(r['passed'] for r in app['performance']['gates'])
    assert current['reference_provenance_verified'] and shared['reference_provenance_verified']
    return products


def previous_closed():
    verify_scope();reports={}
    for label,folder in PRIOR.items():
        assert pin(folder/'closed.json')['sha256']==DIGESTS[label]
        proof=read(folder/'closed.json');assert proof['passed'] and proof['analysis']==pin(folder/'analysis.json')
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
        reports[label]=read(folder/'analysis.json')
    assert read(APP/'closed.json')['admitted']
    products=validate(reports)
    for role in ['selected','candidate']:
        for name,wanted in products[role].items():assert pin(PRODUCT/'collected/runtimes'/role/name)==wanted
    return products


def prepare():
    assert not BASE.exists();products=previous_closed()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals=verify_scope()
    def copy(original,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
        originals[original.relative_to(ROOT).as_posix()]=pin(original)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','candidate_protocol.py',
                 'qualify_outputs.py','identity_scope.py','identity_probes.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for name in ['Program.cs','NpySupport.cs','GraphQualification.csproj']:
        copy(MODEL/'consumer'/name,bundle/'consumer'/name)
    copy(MODEL/'consumer/Program.cs',bundle/'evidence/original-consumer.cs')
    (bundle/'consumer/Program.cs').write_bytes(source((MODEL/'consumer/Program.cs').read_bytes()))
    for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']:
        copy(MODEL/'bridge'/name,bundle/'bridge'/name)
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json','payload.json']:
            copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    copy(PRODUCT/'collected/evidence/compatibility.json',bundle/'evidence/product-compatibility.json')
    copy(MODEL/'evidence/original-manifest.json',bundle/'evidence/original-manifest.json')
    copy(APP_PAYLOAD/'graph-reference.json',bundle/'graph-reference.json')
    copy(TOOLS/'README.md',bundle/'prospective-pyannote.md')
    shutil.copy2(ROOT/'PLAN.md',bundle/'prospective-plan.md')
    copy(ROOT/'global.json',bundle/'global.json')
    stage=dict(passed=True,identities=products,reference_consumer=pin(CURRENT/'collected/built/GraphQualification.dll'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
