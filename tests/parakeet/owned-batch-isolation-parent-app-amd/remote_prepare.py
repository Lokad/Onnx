"""Reuse the exact qualified products and existing complete managed/ORT consumers."""
import json
import os
import shutil
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
from checks import prereqs

BASE=Path(__file__).resolve().parents[1]
CURRENT=Path('/dev/shm/lokad-parakeet-winograd-baseline-20260923')
MODELS=Path('/dev/shm/lokad-parakeet-owned-batch-isolation-models-20260925')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for label,folder in [('baseline',CURRENT),('models',MODELS)]:
        assert pin(folder/'payload.json')==pin(BASE/'evidence'/label/'payload.json')
        assert pin(folder/'collection.json')==pin(BASE/'evidence'/label/'collection.json')
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name
    state=read(MODELS/'identity.json')
    assert state['complete'] and state['code']==0
    assert all(r['complete'] and r['code']==0 for r in state['runs'])
    original=read(CURRENT/'payload.json')
    for name,wanted in original['external'].items():assert pin(name)==wanted,name
    for name in ['assets','runtime']:shutil.copytree(CURRENT/name,BASE/name,copy_function=os.link)
    (BASE/'manifests').mkdir();(BASE/'runtimes').mkdir()
    for role,source in [('current','selected'),('candidate','candidate')]:
        shutil.copytree(MODELS/'runtimes'/source,BASE/'runtimes'/role,copy_function=os.link)
        shutil.copy2(BASE/'evidence'/(role+'-parakeet.json'),BASE/'manifests'/(role+'-parakeet.json'))
        for name,wanted in stage['identities'][role].items():assert pin(BASE/'runtimes'/role/name)==wanted
        for name,wanted in stage['consumers'].items():assert pin(BASE/'runtimes'/role/(name+'.dll'))==wanted
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,previous_owner=state['supervisor'],
        identities=stage['identities'],consumers=stage['consumers'],prerequisites=stage['prerequisites'],
        failed_graph_cases=stage['failed_graph_cases'],release_admitted=False,
        external=original['external'],interpreter=original['interpreter'],python_paths=original['python_paths'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='One causal change: depthwise parent,relocation,ORT,ORT,relocation,depthwise parent;20clips;480requests;unchanged full-call scoring and correctness. No release admission.')
    prereqs(BASE,payload);save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(payload['external']))))


if __name__=='__main__':main()
