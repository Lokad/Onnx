"""Bind terminal layout/copy qualification and the original model assets."""
import copy
import os
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
LAYOUT=Path('/dev/shm/lokad-parakeet-slice-layout-20260924')
CONTRACTS=Path('/dev/shm/lokad-parakeet-validated-composition-build-20260924')
BUILD=Path('/dev/shm/lokad-parakeet-validated-composition-build-20260924')
CURRENT=Path('/dev/shm/lokad-parakeet-wide-entry-first-use-models-20260923')
PREVIOUS=Path('/dev/shm/lokad-pyannote-winograd-product-parakeet-20260923')


def terminal(folder,kind):
    state=read(folder/(kind+'-state.json'))
    assert state['complete'] and not live(state['supervisor'])
    assert all(not live(dict(pid=int(p),birth=b)) for run in state['runs'] for p,b in run['members'].items())
    return state


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for folder,label,kind in [(LAYOUT,'layout','capture'),(CONTRACTS,'contracts','capture'),(BUILD,'build','build')]:
        assert pin(folder/(kind+'-collection.json'))==pin(BASE/'evidence'/(label+'-collection.json'))
        receipt=read(folder/(kind+'-collection.json'));assert receipt['terminal'] and receipt['code']==0
        assert terminal(folder,kind)['code']==0
        # Verify immutable payload inputs; generated test outputs are retained
        # separately and never overwritten by this new stage.
        for name,wanted in read(folder/'spec.json')['files'].items():assert pin(folder/name)==wanted,name
    assert terminal(BUILD,'capture')['code']==0  # Both actual instruction modes qualified.
    for folder,label in [(CURRENT,'current'),(PREVIOUS,'previous')]:
        assert pin(folder/'collection.json')==pin(BASE/'evidence'/(label+'-collection.json'))
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and not any(live(i) for i in receipt['identities'])
        assert pin(folder/'payload.json')==pin(BASE/'evidence'/(label+'-payload.json'))
        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name
    for label in ['layout','contracts']:
        proof=read(BASE/'evidence'/(label+'-closed.json'))
        assert proof['passed'] and proof['analysis']==pin(BASE/'evidence'/(label+'-analysis.json'))
    assert pin(BUILD/'build-review.json')==pin(BASE/'evidence/build-review.json')
    review=read(BASE/'evidence/build-review.json');assert review['passed']
    assert review['built']==pin(BASE/'evidence/built.json')==pin(BUILD/'built.json')
    assert read(BASE/'evidence/contracts-analysis.json')['core']==stage['identities']['candidate']['Lokad.Onnx.dll']
    assert read(BASE/'evidence/contracts-analysis.json')['data']==stage['identities']['candidate']['Lokad.Onnx.Data.dll']
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
        manifest['product_source']='M66 selected release' if role=='selected' else 'M66 admitted-source composition'
        save(BASE/'manifests'/(role+'-parakeet.json'),manifest)
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,previous_owner=read(CONTRACTS/'capture-state.json')['supervisor'],
        identities=stage['identities'],consumers=stage['consumers'],external=original['external'],interpreter=original['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='784 native arrays and all twenty complete public Parakeet clips per product and instruction mode. Exact selected bits and unchanged native 1e-4 gate. Only DOTNET_EnableAVX512=0 is allowed in 256 mode. No scored performance.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
