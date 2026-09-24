"""Verify complete collections, resource envelopes and every owned process."""
import json
from run import BASE, ORIGINAL, pin, read


def resources(kind, original=False):
    root=ORIGINAL if original else BASE
    folder=root/(kind+'-collected');spec=read(root/'bundle/spec.json')
    receipt=read(folder/(kind+'-collection.json'));transfer=read(root/(kind+'-transfer.json'))
    state=read(folder/(kind+'-state.json'))
    assert transfer['passed'] and transfer['archive']==pin(root/(kind+'-results.tar.gz'))
    assert transfer['collection']==pin(folder/(kind+'-collection.json'))
    assert receipt['terminal'] and receipt['code']==state['code']==(1 if original else 0) and state['complete']
    assert receipt['state']==pin(folder/(kind+'-state.json'))
    assert state['supervisor']==read(root/(kind+'-deployment.json'))
    assert pin(folder/'spec.json')==pin(root/'bundle/spec.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    limits=spec[kind+'_limits'];result=[]
    for run in state['runs']:
        assert run['complete'] and run['code']==(-6 if original and run['name']=='inventory' else 0) and run['seconds']<limits['seconds']
        assert run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
        rows=[json.loads(s) for s in (folder/'logs'/(run['name']+'.resources.jsonl')).read_text().splitlines()]
        assert rows and len(rows)==run['samples']
        for row in rows:
            assert row['seconds']<limits['seconds'] and row['rss']<limits['rss']
            assert row['available']>=spec['minimum_free'] and row['tmpfs']>=spec['minimum_free'] and row['output']<spec['output_limit']
            assert row['rss']==sum(m['rss'] for m in row['members'])
            for member in row['members']:
                assert run['members'][str(member['pid'])]==member['birth']
                assert member['affinity']==[2] and all(t==[2] for t in member['threads'])
        gaps=[rows[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(rows,rows[1:])]+[run['seconds']-rows[-1]['seconds']]
        assert all(0<=g<10 for g in gaps)
        result.append(dict(name=run['name'],samples=len(rows),peak_rss=max(r['rss'] for r in rows),seconds=run['seconds']))
    return result
