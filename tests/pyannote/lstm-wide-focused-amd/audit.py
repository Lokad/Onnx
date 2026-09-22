"""Reconcile every source, exact product reference, test case, resource and owner."""
import json
from run import BASE,ROOT,prepared
from prepare import SOURCE,PROJECT,project_text
from protocol import JOBS,LIMITS,check_sample,pin,read,save
from checks import check_suite


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    payload=read(BASE/'payload.json');collected=BASE/'collected';receipt=read(collected/'collection.json');transfer=read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    source=read(SOURCE/'prepared.json')
    for name,wanted in source['files'].items():
        if name.startswith('source/') and name!=PROJECT:assert pin(BASE/'bundle'/name)==wanted and payload['files'][name]==wanted,name
    assert (BASE/'bundle'/PROJECT).read_text()==project_text((SOURCE/PROJECT).read_text())
    assert payload['files'][PROJECT]==pin(BASE/'bundle'/PROJECT)
    state=read(collected/'identity.json');assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0
    assert state['supervisor']==read(BASE/'deployment.json') and [r['name'] for r in state['runs']]==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for row in state['runs'] for p,b in row['members'].items()]
    resources=[];results={}
    for row in state['runs']:
        name=row['name'];assert row['complete'] and row['code']==payload['expected_exit'][name] and row['seconds']<900
        assert row['preflight']==row['preflight_observations'][-1] and row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(line) for line in (collected/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample);assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        if name.startswith('suite-'):
            result=check_suite(collected/name/'suite.trx',BASE/'bundle/references/ordinary.trx',BASE/'bundle/references/scalar.trx',name.split('-')[1])
            assert result==read(collected/name/'review.json');results[name]=result
        resources.append(dict(name=name,exit_code=row['code'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started']<4*3600
    assert (collected/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    built=read(collected/'built.json');assert built['passed'] and built['product']==payload['product']
    for name,wanted in built['files'].items():assert pin(collected/name)==wanted,name
    for name,wanted in payload['product'].items():assert pin(collected/'backend'/name)==wanted,name
    assert built['consumer']==pin(collected/'backend/Lokad.Onnx.Backend.Tests.dll')
    analysis=dict(passed=True,product=payload['product'],consumer=built['consumer'],results=results,resources=resources,
        current_and_candidate_replay_pending=True,codegen_pending=True,no_performance_measurement=True,root_product_changed=False)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__':main()
