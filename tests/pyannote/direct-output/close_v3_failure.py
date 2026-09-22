"""Close the collected AMD NaN mismatch without issuing a timing verdict."""
import complete_v3
from common import *


def main():
    assert not (BASE / 'failure-closed.json').exists()
    prep=prepared(); collected=BASE / 'collected'; receipt=read(collected / 'collection.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
    for name,wanted in receipt['files'].items(): assert pin(collected / name)==wanted,name
    assert receipt['payload']==prep['payload']
    state=read(collected / 'identity.json'); assert state['complete'] and state['code']==1
    assert len(state['runs'])==1 and state['runs'][0]['name']=='validate'
    run=state['runs'][0]; assert run['complete'] and run['code']==-6
    log=(collected / 'logs/validate.log').read_text()
    assert 'm=32 n=64 k=1 stride=1 start=0 bias=True pattern=special index=18 baseline=7fc12345 candidate=ffc00000' in log
    assert not (collected / 'output').exists() or not list((collected / 'output').iterdir())
    assert run['preflight']['available']>=8*1024**3 and run['preflight']['tmpfs']>=3*1024**3
    samples=[json.loads(line) for line in (collected / 'logs/validate.samples.jsonl').read_text().splitlines()]
    assert len(samples)==run['samples'] and max(s['rss'] for s in samples)==run['peak_rss']
    for s in samples:
        assert s['seconds']<900 and s['rss']<2*1024**3 and s['available']>=1024**3 and s['tmpfs']>=1024**3 and s['artifacts']<=1024**3
        assert s['monitor_affinity']==[0] and s['rss']==sum(p['rss'] for p in s['members'])
        for p in s['members']:
            assert {k:p[k] for k in ['pid','birth']}==run['processes']['target'] and p['affinity']==[2]
            assert p['threads'] and all(t['affinity']==[2] for t in p['threads'])
    files=dict(prep['files'])
    for folder in [collected,BASE / 'logs',BASE / 'output']:
        files.update({rel(p):pin(p) for p in folder.rglob('*') if p.is_file()})
    for p in BASE.iterdir():
        if p.is_file(): files[rel(p)]=pin(p)
    save(BASE / 'failure-closed.json',dict(passed=False,closure_passed=True,files=files,
        reason='AMD normal-path bias NaN payload differs; no timing started',resource_samples=len(samples),peak_rss=run['peak_rss'],
        remote_identities=receipt['identities'],terminal=True,code=1))
    print(dict(closed=pin(BASE / 'failure-closed.json'),samples=len(samples),peak_rss=run['peak_rss']))


if __name__=='__main__': main()
