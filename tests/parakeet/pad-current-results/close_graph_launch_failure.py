"""Preserve the missing-working-directory failure before any model worker existed."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/pad-current-graphs-amd'))
from protocol import pin,read,save
from run import BASE,prepared


def main():
    prepared();assert not (BASE/'closed.json').exists()
    c=BASE/'collected';receipt=read(c/'collection.json');transfer=read(BASE/'collection-transfer.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    state=read(c/'identity.json');assert state['complete'] and state['code']==1
    assert state['supervisor']==read(BASE/'deployment.json')==dict(pid=1126449,birth=1790455885.32)
    row,=state['runs'];assert row['name']=='verify-e5-8tok-current'
    assert row['complete'] and row['code'] is None and row['samples']==0 and row['members']=={} and 'child' not in row
    assert receipt['identities']==[state['supervisor']]
    assert 'FileNotFoundError' in state['error'] and "'/dev/shm/lokad-parakeet-pad-current-graphs-20260926/source'" in state['error']
    assert not (c/row['name']/'output/result.json').exists()
    assert not any(name.startswith('source/') for name in read(BASE/'payload.json')['files'])
    analysis=dict(passed=False,admitted=False,no_inference=True,no_timing=True,worker_created=False,
        reason='The unchanged worker uses cwd=BASE/source, but preparation omitted source/global.json and its directory.',
        repair='Restore the pinned source/global.json input in a new namespace; preserve all worker and product bytes.',
        resources_observed=0,terminal=receipt['identities'],original_error=state['error'])
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=False,admitted=False,no_inference=True,analysis=pin(BASE/'analysis.json'),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closure=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__':main()
