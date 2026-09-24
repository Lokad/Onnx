"""Bind closed packing qualification and unchanged native/public reference assets."""
import copy
import os
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
RESIDENCY=Path('/dev/shm/lokad-parakeet-prepared-recurrence-calls-v2-20260924')
CONTRACTS=Path('/dev/shm/lokad-parakeet-prepared-recurrence-contracts-20260924')
BUILD=Path('/dev/shm/lokad-parakeet-prepared-recurrence-build-20260924')
CURRENT=Path('/dev/shm/lokad-parakeet-wide-entry-first-use-models-20260923')
PREVIOUS=Path('/dev/shm/lokad-pyannote-winograd-product-parakeet-20260923')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for folder,label in [(RESIDENCY,'residency'),(CONTRACTS,'contracts'),(BUILD,'build'),(CURRENT,'current'),(PREVIOUS,'previous')]:
        assert pin(folder/'collection.json')==pin(BASE/'evidence'/(label+'-collection.json'))
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==(1 if label=='build' else 0) and receipt['input_error'] is None and not any(live(i) for i in receipt['identities'])
        assert pin(folder/'payload.json')==pin(BASE/'evidence'/(label+'-payload.json'))
        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name
    review=read(BASE/'evidence/review-closed.json');assert review['passed']
    assert review['original_refusal']==pin(BASE/'evidence/build-closed.json')
    assert review['analysis']==pin(BASE/'evidence/review-analysis.json')
    assert read(BASE/'evidence/review-analysis.json')['passed']
    original=read(PREVIOUS/'payload.json');current=read(CURRENT/'payload.json')
    for name,wanted in original['external'].items():assert pin(name)==wanted,name
    for name in ['assets','parakeet-reference']:shutil.copytree(PREVIOUS/name,BASE/name,copy_function=os.link)
    assert pin(BASE/'parakeet-reference/manifest.json')['sha256']=='3bad7d262b8809b1265c84c8e66d02ee38e7d4cff2d92014448976a9e161103c'
    assert pin(BASE/'evidence/original-manifest.json')==pin(CURRENT/'manifests/candidate-parakeet.json')
    (BASE/'manifests').mkdir();(BASE/'runtimes').mkdir()
    for role in ['selected','candidate']:
        folder=BASE/'runtimes'/role;shutil.copytree(CURRENT/'runtimes/candidate',folder,copy_function=os.link)
        for name,wanted in stage['identities'][role].items():
            assert pin(BASE/'products'/role/name)==wanted
            (folder/name).unlink();os.link(BASE/'products'/role/name,folder/name)
        for name,wanted in stage['consumers'].items():assert pin(folder/(name+'.dll'))==wanted
        manifest=copy.deepcopy(read(BASE/'evidence/original-manifest.json'))
        manifest.update(core_sha256=stage['identities'][role]['Lokad.Onnx.dll']['sha256'],data_sha256=stage['identities'][role]['Lokad.Onnx.Data.dll']['sha256'])
        manifest['product_source']='M64 selected Core672e5f30/Data065b7a7f' if role=='selected' else 'M64 prepared recurrence Core3c23b44a/Datacc37b19e'
        save(BASE/'manifests'/(role+'-parakeet.json'),manifest)
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,previous_owner=read(RESIDENCY/'collection.json')['identities'][0],
        identities=stage['identities'],consumers=stage['consumers'],external=original['external'],interpreter=original['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='784 native arrays and all twenty complete public Parakeet clips per product and instruction mode. Exact selected bits and unchanged native 1e-4 gate. Only DOTNET_EnableAVX512=0 is allowed in 256 mode. No scored performance.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
