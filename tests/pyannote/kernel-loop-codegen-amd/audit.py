"""Retain every JIT tier and independently verify all ordinary graph outputs."""
import json
from run import BASE,prepared
from protocol import JOBS,LIMITS,check_sample,pin,read,save
from checks import check_result
from listings import listings


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    c=BASE/'collected';receipt=read(c/'collection.json');payload=read(BASE/'payload.json');transfer=read(BASE/'collection-transfer.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    assert receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    for name,wanted in read(BASE/'bundle/stage.json')['files'].items():assert payload['files'][name]==pin(BASE/'bundle'/name)==wanted,name
    state=read(c/'identity.json');assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS and state['boot_time']==1789634288.0
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    reports={};code={};resources=[]
    for row in state['runs']:
        role=row['name'];assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (c/'logs'/(role+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        result=read(c/role/'result.json');assert result['pid']==row['child']['pid'] and result['runtime']=='10.0.8'
        reports[role]=check_result(result,role,payload,BASE/'bundle',16)
        code[role]=listings(c/'logs'/(role+'.stdout'))
        choices=[r for r in code[role] if r['method'].startswith('Lokad.Onnx.ConvBlockedSpatial:Kernel512(') and r['tier'].startswith('Tier1') and r['complete_uninterleaved']]
        assert choices and any(r['reductions'] for r in choices)
        resources.append(dict(name=role,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started']<4*3600
    save(BASE/'listings.json',code)
    summaries={role:[{k:r[k] for k in ['method','tier','code_bytes','complete_uninterleaved','line']} | dict(reductions=[{k:v for k,v in b.items() if k!='body'} for b in r['reductions']]) for r in rows] for role,rows in code.items()}
    analysis=dict(passed=True,reports=reports,resources=resources,code=summaries,no_performance_measurement=True,manual_codegen_review_pending=True)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),reports=reports,resources=resources,tiers={k:[dict(tier=r['tier'],bytes=r['code_bytes'],complete=r['complete_uninterleaved']) for r in rows] for k,rows in code.items()})))


if __name__=='__main__':main()
