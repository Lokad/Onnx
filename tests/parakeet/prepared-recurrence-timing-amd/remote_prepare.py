"""Require terminal full-model qualification and link the exact qualified call bank."""
import json
import os
from pathlib import Path
import sys
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE=Path(__file__).resolve().parents[1]
PARENTS=dict(calls=Path('/dev/shm/lokad-parakeet-prepared-recurrence-calls-v2-20260924'),
             models=Path('/dev/shm/lokad-parakeet-prepared-recurrence-models-20260924'))


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    external={};parents={}
    for label,folder in PARENTS.items():
        for name in ['collection.json','payload.json']:assert pin(folder/name)==pin(BASE/'evidence'/label/name)
        receipt=read(folder/'collection.json');proof=read(BASE/'evidence'/label/'closed.json')
        assert proof['passed'] and receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        assert proof['analysis']==pin(BASE/'evidence'/label/'analysis.json')
        assert read(BASE/'evidence'/label/'analysis.json')['passed']
        parent=read(folder/'payload.json');parents[label]=parent
        for name,wanted in parent['files'].items():assert pin(folder/name)==wanted,name
        for name,wanted in parent['external'].items():
            assert name not in external or external[name]==wanted;external[name]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    assert parents['calls']['interpreter']==parents['models']['interpreter']==pin(Path(sys.executable))
    for name,link in stage['links'].items():
        source=(PARENTS['calls']/link['source']).resolve();target=(BASE/name).resolve()
        assert source.is_relative_to(PARENTS['calls']) and target.is_relative_to(BASE)
        assert pin(source)==link['pin'];target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,identities=stage['identities'],boot_time=1789634288.0,
        previous_owner=read(PARENTS['models']/'collection.json')['identities'][0],feed=parents['calls']['feed'],
        external=external,interpreter=parents['calls']['interpreter'],python_paths=parents['calls']['python_paths'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='All 380 actual complete LSTM calls, five warmup/five measured passes, duplicate products in both modes; preparation costs retained separately.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
