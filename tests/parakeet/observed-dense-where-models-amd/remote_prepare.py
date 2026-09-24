"""Require terminal build/numerical qualification and reuse the original model assets."""
import copy
import os
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
BUILD=Path('/dev/shm/lokad-parakeet-observed-dense-where-inventory-20260924')
CONTRACTS=Path('/dev/shm/lokad-parakeet-observed-dense-where-numerics-20260924')
CURRENT=Path('/dev/shm/lokad-parakeet-validated-composition-models-20260924')
PREVIOUS=Path('/dev/shm/lokad-pyannote-winograd-product-parakeet-20260923')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for folder,label in [(CURRENT,'current'),(PREVIOUS,'previous'),(BUILD,'build'),(CONTRACTS,'contracts')]:
        assert pin(folder/'collection.json')==pin(BASE/'evidence'/(label+'-collection.json'))
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        assert pin(folder/'payload.json')==pin(BASE/'evidence'/(label+'-payload.json'))
        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name
    for label in ['build','contracts']:
        proof=read(BASE/'evidence'/(label+'-closed.json'))
        assert proof['passed'] and proof['analysis']==pin(BASE/'evidence'/(label+'-analysis.json'))
    qualification=read(BASE/'evidence/contracts-analysis.json')['products']
    assert stage['identities']==dict(selected=qualification['current'],candidate=qualification['candidate'])
    original=read(PREVIOUS/'payload.json')
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
        manifest['product_source']='M70 current release' if role=='selected' else 'M70 observed-mask composition'
        save(BASE/'manifests'/(role+'-parakeet.json'),manifest)
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,
        previous_owner=read(CONTRACTS/'collection.json')['identities'][0],
        identities=stage['identities'],consumers=stage['consumers'],external=original['external'],interpreter=original['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='784 arrays against pinned ORT reference and twenty full public clips per product/instruction mode. Exact current bits and unchanged native 1e-4 gate. No scored performance.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
