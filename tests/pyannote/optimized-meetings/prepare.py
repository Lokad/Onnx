"""Bind a qualified optimized core to the unchanged two-meeting public consumer."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-optimized-meetings-20260921'
PRODUCT=ROOT/'artifacts/pyannote-lstm-output-lanes-20260921'
ORIGINAL=ROOT/'artifacts/pyannote-natural-meetings-20260920'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def clean_env():return {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}


def main():
    old_receipt=ORIGINAL/'closed.json'
    assert pin(old_receipt)['sha256']=='3ada9904a98aa61ee7fd03db5ce65db206c8278fc13a0939a0ce20249fd5100d'
    old=json.loads(old_receipt.read_text())
    assert old['closed'] and old['execution_passed'] and old['public_comparison_passed'] and old['all_owned_processes_terminal']
    product_receipt=PRODUCT/'qualification-closed.json';product=json.loads(product_receipt.read_text());assert product['passed']
    assert pin(PRODUCT/'manifest.json')['sha256']=='db65491656c9ca8e1a35def29046d8cec3d4ed94add5ba11f730ab65a12582a3'
    for name,wanted in old['files'].items():assert pin(ORIGINAL/name)==wanted,name
    for name,wanted in old['sources'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in product['files'].items():assert pin(ROOT/name)==wanted,name
    BASE.mkdir(exist_ok=False)
    for name in ['bin','inputs','prior','logs']:(BASE/name).mkdir()
    for name in ['NaturalMeetings.dll','NaturalMeetings.deps.json','NaturalMeetings.runtimeconfig.json']:
        shutil.copy2(ORIGINAL/'bin'/name,BASE/'bin'/name)
    assert pin(BASE/'bin/NaturalMeetings.dll')['sha256']=='79e3e7990ba6aa29e42da788277aad41b774ff3b8c3966b18ab1101944d0c0f1'
    for path in (PRODUCT/'runtimes/candidate').glob('*.dll'):shutil.copy2(path,BASE/'bin'/path.name)
    for name in ['dataset.json','ES2004a-600s.wav','IS1009a-600s.wav']:shutil.copy2(ORIGINAL/'inputs'/name,BASE/'inputs'/name)
    for name,target in [('manifest.json','manifest.json'),('audit.json','scores.json'),('process-native-run/worker/result.json','native.json')]:
        shutil.copy2(ORIGINAL/name,BASE/'prior'/target)
    manifest=json.loads((ORIGINAL/'manifest.json').read_text())
    manifest.update(core_sha256=pin(BASE/'bin/Lokad.Onnx.dll')['sha256'],data_sha256=pin(BASE/'bin/Lokad.Onnx.Data.dll')['sha256'],
        accuracy_scope='Qualified isolated spatial-copy/LSTM core; original two meetings and recovery; no native timing replay')
    assert manifest['core_sha256']=='469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd'
    assert manifest['data_sha256']=='e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'
    manifest['limits']=dict(seconds=3600,rss=8*1024**3,available=1024**3,preflight=10*1024**3,disk=20*1024**3,preflight_wait_seconds=3600)
    save(BASE/'manifest.json',manifest)
    files={str(old_receipt.relative_to(ROOT)):pin(old_receipt),str(product_receipt.relative_to(ROOT)):pin(product_receipt)}
    for path in BASE.rglob('*'):
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    for item in manifest['models'].values():
        assert pin(ROOT/item['path'])=={k:item[k] for k in ('bytes','sha256')}
        files[item['path']]=pin(ROOT/item['path'])
    for name in ['audit.py','common.py']:files['tests/pyannote/natural-meetings/'+name]=pin(ROOT/'tests/pyannote/natural-meetings'/name)
    save(BASE/'prepared.json',dict(files=files,manifest=pin(BASE/'manifest.json'),product_receipt=pin(product_receipt),prior_receipt=pin(old_receipt)))
    print(json.dumps(dict(files=len(files),prepared=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
