"""Bind unchanged selected products, saved trajectories, SDK/feed and native libraries."""
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE=Path(__file__).resolve().parents[1]
PARENTS={k:Path('/dev/shm/lokad-parakeet-inclusive-packing-'+k+'-20260924') for k in ['models','app','build']}


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    external={};parents={}
    for label,folder in PARENTS.items():
        for name in ['collection.json','payload.json']:assert pin(folder/name)==pin(BASE/'evidence'/label/name)
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        parent=read(folder/'payload.json');parents[label]=parent
        for name,wanted in parent['files'].items():assert pin(folder/name)==wanted,name
        for name,wanted in parent['external'].items():
            assert name not in external or external[name]==wanted
            external[name]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    assert parents['app']['interpreter']==parents['build']['interpreter']==pin(Path(__import__('sys').executable))
    arrays=BASE/'selected-arrays';arrays.mkdir()
    for name,wanted in stage['arrays'].items():
        source=PARENTS['models']/'selected-native-512/result.json.tensors'/name
        assert pin(source)==wanted and Path(name).name==name;os.link(source,arrays/name)
    assert pin(stage['model_path'])==stage['model'];external[stage['model_path']]=stage['model']
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,identities=stage['identities'],boot_time=1789634288.0,
        previous_owner=read(PARENTS['app']/'collection.json')['identities'][0],feed=parents['build']['feed'],
        external=external,interpreter=parents['app']['interpreter'],python_paths=parents['app']['python_paths'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='Selected original decoder capture and native complete-call correctness only; no candidate, encoder execution or performance measurement.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
