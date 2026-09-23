"""Reconcile all finite-extreme calls and exact current/candidate output hashes."""
import json
from run import BASE,prepared
from protocol import JOBS,LIMITS,pin,read,save,check_sample
from checks import check_result

def main():
    spec=prepared();assert not (BASE/'closed.json').exists();c=BASE/'collected'
    receipt=read(c/'collection.json');transfer=read(BASE/'collection-transfer.json');payload=read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    built=read(c/'built.json');assert built['passed']
    for name,wanted in built['files'].items():assert pin(c/name)==wanted,name
    for role,product in payload['products'].items():
        for name,wanted in product.items():assert pin(c/'runtime'/role/name)==wanted,name
        assert pin(c/'runtime'/role/'Lokad.Onnx.Backend.Tests.dll')==payload['probe']
        assert pin(c/'runtime'/role/'ConvExtremes.dll')==built['consumer']
    state=read(c/'identity.json');assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json') and state['boot_time']==1789634288.0
    assert [r['name'] for r in state['runs']]==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[];results={};raw={};actual=dict(payload,consumer=built['consumer'])
    for row in state['runs']:
        name=row['name'];assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']==row['preflight_observations'][-1] and row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(line) for line in (c/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample);assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        if name.startswith(('selected-','candidate-')):
            value=read(c/name/'result.json');assert value['pid']==row['child']['pid']
            results[name]=check_result(value,name,actual);assert results[name]==read(c/name/'review.json');raw[name]=value
        resources.append(dict(name=name,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert (c/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204') and state['ended']-state['started']<4*3600
    for width in ['256','512']:
        a,b=raw['selected-'+width],raw['candidate-'+width]
        assert a['observations']==b['observations'] and a['graph_cases']==b['graph_cases']
    analysis=dict(passed=True,products=payload['products'],probe=payload['probe'],consumer=built['consumer'],
        results=results,resources=resources,all_current_candidate_hashes_exact=True,graph_checks_exact=True,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))

if __name__=='__main__':main()
