"""Bind closed captures and reviewed products; hardlink only immutable input files."""
import json
import os
from pathlib import Path
import sys
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE=Path(__file__).resolve().parents[1]
PARENTS=dict(capture=Path('/dev/shm/lokad-parakeet-decoder-lstm-capture-20260924'),
             build=Path('/dev/shm/lokad-parakeet-prepared-recurrence-build-20260924'),
             contracts=Path('/dev/shm/lokad-parakeet-prepared-recurrence-contracts-20260924'))


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    external={};parents={}
    for label,folder in PARENTS.items():
        for name in ['collection.json','payload.json']:assert pin(folder/name)==pin(BASE/'evidence'/label/name)
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==(1 if label=='build' else 0) and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        if label=='build':
            review=read(BASE/'evidence/review/closed.json');assert review['passed']
            assert review['original_refusal']==pin(BASE/'evidence/build/closed.json')
            assert review['analysis']==pin(BASE/'evidence/review/analysis.json')
            assert read(BASE/'evidence/review/analysis.json')['passed']
        parent=read(folder/'payload.json');parents[label]=parent
        for name,wanted in parent['files'].items():assert pin(folder/name)==wanted,name
        for name,wanted in parent['external'].items():
            assert name not in external or external[name]==wanted;external[name]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    assert parents['capture']['interpreter']==parents['build']['interpreter']==pin(Path(sys.executable))
    for name,link in stage['links'].items():
        source=(PARENTS['capture']/link['source']).resolve();target=(BASE/name).resolve()
        assert source.is_relative_to(PARENTS['capture']) and target.is_relative_to(BASE)
        assert pin(source)==link['pin'];target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    assert pin(stage['model_path'])==stage['model'];external[stage['model_path']]=stage['model']
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,identities=stage['identities'],boot_time=1789634288.0,
        previous_owner=read(PARENTS['contracts']/'collection.json')['identities'][0],feed=parents['capture']['feed'],
        external=external,interpreter=parents['capture']['interpreter'],python_paths=parents['capture']['python_paths'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='Actual decoder residency, route and all captured complete calls in both modes; no encoder inference or performance measurement.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()
