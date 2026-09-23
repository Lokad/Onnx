"""Freeze fresh complete Pyannote qualification of the admitted M43 product."""
import ast, importlib.util, json, shutil, tarfile
from pathlib import Path
from protocol import pin, read, save

ROOT=Path(__file__).resolve().parents[3]; TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-first-use-kernels-pyannote-amd-20260923'
CURRENT=ROOT/'artifacts/pyannote-winograd-product-models-amd-v2-20260923'
MODEL=CURRENT/'bundle'; APP_PAYLOAD=CURRENT/'collected'
PRODUCT=ROOT/'artifacts/parakeet-first-use-kernels-build-amd-20260923'
APP=ROOT/'artifacts/parakeet-first-use-kernels-app-amd-20260923'
OLD_DATA='f3b9aa81ee9766797e95216714dec559c5d8b020f8df510cf5f7ee0dda82693a'
NEW_DATA='9b623be98cfab7f00601fc56e264c20ae8da3d40dfcd9ab1091e741e3f85dcd0'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('models_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)

def previous_closed():
    for folder, digest in [(CURRENT,'82021de9672fb54cc684cc03c35e3d8c84ffb7b180dfcba7fa2c8a03f3296865'),
                           (PRODUCT,'2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json'); assert proof['passed']
        for name,wanted in proof['files'].items(): assert pin(folder/name)==wanted,name
    proof=read(APP/'closed.json'); assert proof['passed'] and proof['admitted']
    for name,wanted in proof['files'].items(): assert pin(APP/name)==wanted,name

def prepare():
    assert not BASE.exists(); previous_closed()
    for name in ['candidate_protocol.py','qualify_outputs.py']:
        assert (TOOLS/name).read_bytes()==(ROOT/'tests/pyannote/blocked-spatial-app-amd'/name).read_bytes()
    BASE.mkdir(); bundle=BASE/'bundle'; bundle.mkdir(); originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','candidate_protocol.py','qualify_outputs.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for name in ['Program.cs','NpySupport.cs','GraphQualification.csproj']: copy(MODEL/'consumer'/name,bundle/'consumer'/name)
    source=bundle/'consumer/Program.cs'; before=source.read_bytes(); assert before.count(OLD_DATA.encode())==1
    copy(MODEL/'consumer/Program.cs',bundle/'evidence/original-consumer.cs')
    source.write_bytes(before.replace(OLD_DATA.encode(),NEW_DATA.encode()))
    assert source.read_bytes().replace(NEW_DATA.encode(),OLD_DATA.encode())==before
    for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']: copy(MODEL/'bridge'/name,bundle/'bridge'/name)
    for label,folder in [('current',CURRENT),('product',PRODUCT),('app',APP)]:
        for name in ['closed.json','analysis.json','payload.json']: copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    copy(MODEL/'evidence/original-manifest.json',bundle/'evidence/original-manifest.json')
    copy(APP_PAYLOAD/'graph-reference.json',bundle/'graph-reference.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    copy(ROOT/'global.json',bundle/'global.json')
    stage=dict(passed=True,current_collection=pin(CURRENT/'collected/collection.json'),current_payload=pin(CURRENT/'payload.json'),
        selected_product=read(CURRENT/'analysis.json')['identities']['candidate'],
        product=read(PRODUCT/'analysis.json')['built'],selected_consumer=pin(CURRENT/'collected/built/GraphQualification.dll'),
        old_data=OLD_DATA,new_data=NEW_DATA,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert stage['product']['Lokad.Onnx.Data.dll']['sha256']==NEW_DATA
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file(): originals[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))

if __name__=='__main__': prepare()
