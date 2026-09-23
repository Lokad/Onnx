"""Freeze a literal-only consumer adaptation and reuse closed model assets on AMD."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-winograd-product-models-amd-v2-20260923'
PRODUCT=ROOT/'artifacts/pyannote-winograd-product-amd-20260923'
APP=ROOT/'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922'
APP_PAYLOAD=ROOT/'artifacts/pyannote-blocked-spatial-app-amd-payload-20260922/payload'
CURRENT=ROOT/'artifacts/pyannote-lstm-input-models-amd-20260922'
MODEL=CURRENT/'bundle'
BRIDGE=ROOT/'artifacts/pyannote-combined-consumers-20260922'
OLD_DATA='b935837024c3ffd67e322ba1fe44405b24b7da221779cf00856f090df08693f5'
NEW_DATA='f3b9aa81ee9766797e95216714dec559c5d8b020f8df510cf5f7ee0dda82693a'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('models_monitor',MONITOR);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    for folder,digest,relative in [
        (PRODUCT,'399dde04df28a204d83e4dcb27ed92c7f013af28343994f7e2f42539ec593b17',False),
        (APP,'5c238cd33845eb58fc00332361a967185ae82a9fcb70854530e07fad58f064d0',False),
        (CURRENT,'2a348f67238a44e3cc36f58e8cd5bf467fa1a8562c57c788a04902f8bdb8c90b',False),
        (BRIDGE,'602d752956d6cf830851cd0dc35eb4bdf1230880fb2dce6d8bbea6f530bb555c',True)]:
        assert pin(folder/'closed.json')['sha256']==digest
        for name,wanted in read(folder/'closed.json')['files'].items():assert pin((ROOT if relative else folder)/name)==wanted,name
    assert pin(APP_PAYLOAD/'payload.json')['sha256']=='229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f'


def prepare():
    assert not BASE.exists();previous_closed()
    assert (TOOLS/'candidate_protocol.py').read_bytes()==(ROOT/'tests/pyannote/blocked-spatial-app-amd/candidate_protocol.py').read_bytes()
    original_audit=(ROOT/'tests/pyannote/blocked-spatial-app-amd/qualify_outputs.py').read_text()
    assert (TOOLS/'qualify_outputs.py').read_text()==original_audit.replace("all(c['failed_values'] == 0 for c in comparisons)","all(c['failed_values'] == 0 for c in comparisons if c['reference'] == 'native')")
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','candidate_protocol.py','qualify_outputs.py']:copy(TOOLS/name,bundle/'tools'/name)
    for name in ['Program.cs','NpySupport.cs','GraphQualification.csproj']:copy(MODEL/'consumer'/name,bundle/'consumer'/name)
    source=bundle/'consumer/Program.cs';before=source.read_bytes();assert before.count(OLD_DATA.encode())==1
    copy(MODEL/'consumer/Program.cs',bundle/'evidence/original-consumer.cs')
    source.write_bytes(before.replace(OLD_DATA.encode(),NEW_DATA.encode()))
    assert source.read_bytes().replace(NEW_DATA.encode(),OLD_DATA.encode())==before
    for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']:copy(BRIDGE/'bridge/bin/Release/net10.0'/name,bundle/'bridge'/name)
    for name in ['analysis.json','closed.json']:copy(PRODUCT/name,bundle/'evidence'/('product-'+name))
    copy(APP_PAYLOAD/'manifests/portable-pyannote.json',bundle/'evidence/original-manifest.json')
    copy(APP_PAYLOAD/'graph-reference.json',bundle/'graph-reference.json')
    copy(ROOT/'PLAN.md',bundle/'prospective-plan.md');originals.pop('PLAN.md')
    for p in [APP_PAYLOAD/'manifests/portable-pyannote.json',APP_PAYLOAD/'graph-reference.json']:
        assert pin(p)==read(APP_PAYLOAD/'payload.json')['files'][p.relative_to(APP_PAYLOAD).as_posix()]
    copy(PRODUCT/'bundle/source/global.json',bundle/'global.json')
    stage=dict(passed=True,current_collection=pin(CURRENT/'collected/collection.json'),current_payload=pin(CURRENT/'payload.json'),selected_product=read(CURRENT/'analysis.json')['identities']['candidate'],product=read(PRODUCT/'analysis.json')['measured'],selected_consumer=pin(CURRENT/'collected/built/GraphQualification.dll'),
        old_data=OLD_DATA,new_data=NEW_DATA,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    files=dict(originals)
    for p in [*TOOLS.iterdir(),MONITOR,*[folder/'closed.json' for folder in [PRODUCT,APP,CURRENT,BRIDGE]],APP_PAYLOAD/'payload.json']:
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
