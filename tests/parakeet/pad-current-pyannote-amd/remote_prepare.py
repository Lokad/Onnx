"""Hardlink immutable model assets and reviewed products; build no product here."""
import copy
import json
import os
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
from identity_scope import source

BASE=Path(__file__).resolve().parents[1]
CURRENT=Path('/dev/shm/lokad-parakeet-observed-dense-where-pyannote-20260924')
PRODUCT=Path('/dev/shm/lokad-parakeet-pad-current-models-20260926')
APP=Path('/dev/shm/lokad-parakeet-pad-current-app-20260926')
SHARED=Path('/dev/shm/lokad-parakeet-pad-current-shared-20260926')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for folder,label in [(CURRENT,'current'),(PRODUCT,'product'),(APP,'app'),(SHARED,'shared')]:
        assert pin(folder/'payload.json')==pin(BASE/'evidence'/(label+'-payload.json'))
        assert pin(folder/'collection.json')==pin(BASE/'evidence'/(label+'-collection.json'))
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name
    assert read(BASE/'evidence/app-closed.json')['admitted']
    assert stage['identities']==read(BASE/'evidence/shared-analysis.json')['identities']==read(PRODUCT/'payload.json')['identities']
    current=read(CURRENT/'payload.json');built=read(CURRENT/'built.json')
    assert pin(CURRENT/'built.json')==read(CURRENT/'collection.json')['files']['built.json']
    for name,wanted in built['files'].items():assert pin(CURRENT/name)==wanted,name
    external=dict(read(PRODUCT/'payload.json')['external'])
    for name,wanted in current['external'].items():
        assert name not in external or external[name]==wanted;external[name]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    for name in ['assets','graph-reference']:shutil.copytree(CURRENT/name,BASE/name,copy_function=os.link)
    assert pin(BASE/'graph-reference.json')==current['files']['graph-reference.json']
    shutil.copytree(CURRENT/'runtimes/candidate',BASE/'reference-runtime',copy_function=os.link)
    assert pin(BASE/'reference-runtime/GraphQualification.dll')==stage['reference_consumer']
    (BASE/'manifests').mkdir();(BASE/'runtimes').mkdir()
    for role in ['selected','candidate']:
        folder=BASE/'runtimes'/role
        shutil.copytree(CURRENT/'runtimes/candidate',folder,copy_function=os.link)
        for name,wanted in stage['identities'][role].items():
            original=PRODUCT/'runtimes'/role/name;assert pin(original)==wanted
            (folder/name).unlink();os.link(original,folder/name)
        for suffix in ['dll','deps.json','runtimeconfig.json']:(folder/('GraphQualification.'+suffix)).unlink()
        manifest=copy.deepcopy(read(BASE/'evidence/original-manifest.json'))
        assert manifest==read(CURRENT/'evidence/original-manifest.json')
        manifest.update(core_sha256=stage['identities'][role]['Lokad.Onnx.dll']['sha256'],
            data_sha256=stage['identities'][role]['Lokad.Onnx.Data.dll']['sha256'])
        manifest['product_source']='Qualified current root' if role=='selected' else 'Current-root padding dispatcher'
        save(BASE/'manifests'/(role+'-pyannote.json'),manifest)
    assert (BASE/'consumer/Program.cs').read_bytes()==source((BASE/'evidence/original-consumer.cs').read_bytes())
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,
        previous_owner=read(SHARED/'collection.json')['identities'][0],identities=stage['identities'],
        reference_consumer=stage['reference_consumer'],external=external,interpreter=read(PRODUCT/'payload.json')['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='One common identity-parameterized consumer;18 arrays and16 complete public calls per actual product;original numerical and ownership gates;no score.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(external))))


if __name__=='__main__':main()
