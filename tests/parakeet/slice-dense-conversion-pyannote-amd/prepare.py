"""Freeze fresh complete Pyannote qualification of the admitted M73 product."""
import ast, importlib.util, json, shutil, tarfile
from pathlib import Path
from protocol import pin, read, save
from consumer_scope import verify_scope

ROOT=Path(__file__).resolve().parents[3]; TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-pyannote-amd-20260925'
CURRENT=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-amd-20260924'
MODEL=CURRENT/'bundle'; APP_PAYLOAD=CURRENT/'collected'
PRODUCT=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
APP=ROOT/'artifacts/parakeet-slice-dense-conversion-app-amd-20260925'
NEW_DATA='a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('models_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)

def previous_closed():
    verify_scope()
    assert pin(PRODUCT/'closed.json')['sha256']=='5860fc366c3d976907f35f8d7a1832c10566fa08910af32c168dca7a63979842'
    assert pin(CURRENT/'closed.json')['sha256']=='e36fe9c83405608659ebdc4d673c185c8805a5414a6b01d401904209d59417dd'
    for folder in [CURRENT,PRODUCT]:
        proof=read(folder/'closed.json'); assert proof['passed']
        assert proof['analysis']==pin(folder/'analysis.json')
        for name,wanted in proof['files'].items(): assert pin(folder/name)==wanted,name
    identities=read(PRODUCT/'analysis.json')['identities']
    assert identities['selected']==read(CURRENT/'analysis.json')['identities']['candidate']
    assert identities['selected']['Lokad.Onnx.Data.dll']==identities['candidate']['Lokad.Onnx.Data.dll']
    from checks import inventory
    built=read(CURRENT/'collected/built.json')
    scope=inventory(read(CURRENT/'collected/consumer-inventory/instructions.json'),read(CURRENT/'payload.json'),built)
    assert scope==read(CURRENT/'analysis.json')['inventory']==read(CURRENT/'collected/consumer-inventory/review.json')
    assert built['consumer']==read(CURRENT/'analysis.json')['consumers']['candidate']==pin(CURRENT/'collected/built/GraphQualification.dll')
    assert built['consumer']['sha256']=='94210bcb068b68b16bc14c7194381deae6f98be88787365c0250aa96862061e3'
    assert (MODEL/'consumer/Program.cs').read_text().count(NEW_DATA)==1
    assert read(APP/'analysis.json')['identities']==dict(current=identities['selected'],candidate=identities['candidate'])
    proof=read(APP/'closed.json'); assert proof['passed'] and proof['admitted']
    for name,wanted in proof['files'].items(): assert pin(APP/name)==wanted,name

def prepare():
    assert not BASE.exists(); previous_closed()
    for name in ['candidate_protocol.py','qualify_outputs.py']:
        assert (TOOLS/name).read_bytes()==(ROOT/'tests/pyannote/blocked-spatial-app-amd'/name).read_bytes()
    BASE.mkdir(); bundle=BASE/'bundle'; bundle.mkdir(); originals=verify_scope()
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','candidate_protocol.py','qualify_outputs.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for name in ['Program.cs','NpySupport.cs','GraphQualification.csproj']: copy(MODEL/'consumer'/name,bundle/'consumer'/name)
    copy(MODEL/'consumer/Program.cs',bundle/'evidence/original-consumer.cs')
    for name in ['instructions.json','review.json']:copy(CURRENT/'collected/consumer-inventory'/name,bundle/'evidence/consumer-inventory'/name)
    copy(CURRENT/'collected/built.json',bundle/'evidence/consumer-built.json')
    for label,folder in [('current',CURRENT),('product',PRODUCT),('app',APP)]:
        for name in ['closed.json','analysis.json','payload.json']: copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    copy(MODEL/'evidence/original-manifest.json',bundle/'evidence/original-manifest.json')
    copy(APP_PAYLOAD/'graph-reference.json',bundle/'graph-reference.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    copy(ROOT/'global.json',bundle/'global.json')
    stage=dict(passed=True,current_collection=pin(CURRENT/'collected/collection.json'),current_payload=pin(CURRENT/'payload.json'),
        selected_product=read(CURRENT/'analysis.json')['identities']['candidate'],
        product=read(PRODUCT/'analysis.json')['identities']['candidate'],selected_consumer=pin(CURRENT/'collected/built/GraphQualification.dll'),
        no_rebuild=True,new_data=NEW_DATA,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
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
