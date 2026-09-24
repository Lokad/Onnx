"""Audit complete numerical census, cross-product identities, codegen and resources."""
import json
import re
import numpy as np

from protocol import JOBS,LIMITS,pin,read,save,check_sample
from prepare import ROOT,BASE,previous_closed

def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    folder=BASE/'collected';receipt=read(folder/'collection.json');state=read(folder/'identity.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json')
    payload=read(BASE/'payload.json')
    transfer=read(BASE/'collection-transfer.json');assert transfer['passed']
    assert transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(folder/'collection.json')
    assert prepared['archive']==pin(BASE/'payload.tar.gz') and prepared['stage']==pin(BASE/'bundle/stage.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0
    assert state['supervisor']==read(BASE/'deployment.json')
    assert state['ended']-state['started']<4*3600
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=0;peak=0;identities={(v['pid'],v['birth']) for v in receipt['identities']}
    assert (state['supervisor']['pid'],state['supervisor']['birth']) in identities
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        limit=LIMITS['build_preflight_available' if row['name'] in JOBS[:3] else 'preflight_available']
        assert row['preflight']['available']>=limit and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples'] and samples
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:
                assert (member['pid'],member['birth']) in identities
                assert row['members'][str(member['pid'])]==member['birth']
        assert max(s['rss'] for s in samples)==row['peak_rss']
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources+=len(samples);peak=max(peak,row['peak_rss'])
    built=read(folder/'built.json');assert built['passed']
    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items(): assert pin(folder/'runtimes'/role/name)==wanted
    census=read(BASE/'bundle/cases.json'); assert len(census)==8
    for name,wanted in read(BASE/'bundle/stage.json')['files'].items():
        assert payload['files'][name]==wanted and pin(BASE/'bundle'/name)==wanted,name
    results={}
    for run in state['runs'][3:]:
        name=run['name'];role,mode,width=name.split('-');v=read(folder/name/'result.json')
        assert v['completed'] and v['no_performance_measurement'] and v['runtime']=='10.0.8'
        assert v['role']==role and v['mode']==mode=='probe' and v['width']==int(width)
        assert v['assembly']==built['consumer']['sha256'] and v['pid']==run['child']['pid']
        assert v['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']
        assert v['flags']==({} if width=='512' else {'DOTNET_EnableAVX512':'0'})
        assert len(v['rows'])==8
        for row,case in zip(v['rows'],census,strict=True):
            assert row['name']==case['name'] and row['mask']==case['mask'] and row['inputs_unchanged']
            expected=np.where(np.array(case['mask'])!=0,np.float32(-10000),np.array([10,20],dtype=np.float32)).view(np.int32).tolist()
            assert row['expected']==expected
            assert row['matches_nonzero_selection']==(row['actual']==expected)
        results[name]=v['rows']
    for role in ['current','candidate']:assert results[role+'-probe-256']==results[role+'-probe-512']
    assert all(r['matches_nonzero_selection'] for r in results['current-probe-256'])
    failures=[r for r in results['candidate-probe-256'] if not r['matches_nonzero_selection']]
    analysis=dict(passed=True,diagnostic_only=True,candidate_numerically_admitted=False,
        all_current_cases_match=True,candidate_failures=failures,results=results,
        resources=resources,peak_rss=peak,consumer=built['consumer'],products=payload['products'],
        root_product_changed=False,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,diagnostic_only=True,candidate_numerically_admitted=False,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),failures=failures,resources=resources,peak_rss=peak)))


if __name__=='__main__':main()
