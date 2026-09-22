"""Retain every clock and evaluate the unchanged complete-call screen gates."""
import importlib.util
import json
from run import ROOT, BASE, prepared
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from checks import check_result
from score import ORDER, validate_and_score


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    tests=read(BASE/'selftest.json');assert tests['passed'] and tests['tests']==12
    payload=read(BASE/'payload/payload.json');collected=BASE/'collected';receipt=read(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    transfer=read(BASE/'collection-transfer.json')
    assert transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    for name,wanted in payload['files'].items():assert pin(BASE/'payload'/name)==wanted,name
    module_path=ROOT/'tests/parakeet/portable-models/common.py'
    loader=importlib.util.spec_from_file_location('filter_screen_resources',module_path)
    common=importlib.util.module_from_spec(loader);loader.loader.exec_module(common)
    common.verify(read(BASE/'inputs.json')['files'])
    local=common.resources(BASE,'controller.json',{'restore':(8,8,900,False),'build':(8,8,900,False),
        'local-production':(12,8,900,True),'local-candidate':(12,8,900,True)})
    local_state=read(BASE/'controller.json')
    for role in ['production','candidate']:
        result=read(BASE/'output'/('local-'+role)/'result.json');check_result(result,role,payload,BASE/'payload',8,False)
        row,=[r for r in local_state['runs'] if r['name']=='local-'+role]
        assert result['runtime']=='10.0.12' and result['pid']==row['worker']['pid']
    state=read(collected/'identity.json');assert state['complete'] and state['code']==0
    assert state['supervisor']==read(BASE/'deployment.json') and state['boot_time']==1789634288.0
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS==ORDER
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[];reports={}
    for row in state['runs']:
        name=row['name'];role=name.split('-')[0]
        assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        assert row['preflight']==row['preflight_observations'][-1]==read(collected/(name+'-preflight.json'))[-1]
        samples=[json.loads(s) for s in (collected/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:assert row['members'][str(member['pid'])]==member['birth']
        result=read(collected/name/'result.json');assert result['pid']==row['child']['pid'] and result['runtime']=='10.0.8'
        check_result(result,role,payload,BASE/'payload',16,True)
        journal=[json.loads(s) for s in (collected/name/'journal.jsonl').read_text().splitlines()]
        assert journal==result['preparation']+result['observations'];reports[name]=result
        resources.append(dict(name=name,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started']<4*3600
    analysis=validate_and_score(reports,read(BASE/'payload/fixtures/result.json'),read(BASE/'payload/reference.json'))
    assert analysis['iteration_manifest']==read(BASE/'payload/iterations.json')
    assert analysis['calls']==17184 and analysis['warmups']==4296 and analysis['measured']==12888
    analysis.update(complete=True,numerical_and_resource_checks_pass=True,resources=resources,local_resources=local['resources'],
        samples=sum(r['samples'] for r in resources),peak_rss=max(r['peak_rss'] for r in resources),
        no_application_speed_measurement=True,no_ort_speed_measurement=True,
        boundary='Both roles use unchanged prepared ordinary graph callers; include caller assertions/recording, scans, rentals, conversions and epilogues; exclude hashes/journal IO. Separate preparation clocks include graph creation and preparation.')
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json',dict(passed=True,admitted=analysis['admitted'],files=files,local_inputs=spec['files'],local_identities=local['identities'],remote_terminal=receipt['identities']))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),admitted=analysis['admitted'],aggregate=analysis['rows'][0],
        failed_controls=[r for r in analysis['controls'] if not r['passed']],failed_gates=[r for r in analysis['gates'] if not r['passed']],samples=analysis['samples'],peak_rss=analysis['peak_rss'])))


if __name__=='__main__':main()
