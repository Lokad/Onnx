"""Close all qualified raw clocks, including stable rejection or failed controls."""
import importlib.util
import json
from run import BASE, prepared
from protocol import JOBS, TIMING_JOBS, LIMITS, check_sample, pin, read, save
from checks import qualify, evaluate


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    collected=BASE/'collected';receipt=read(collected/'collection.json');transfer=read(BASE/'collection-transfer.json');payload=read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    stage=read(BASE/'bundle/stage.json')
    for name,wanted in stage['files'].items():
        assert payload['files'][name]==wanted and pin(BASE/'bundle'/name)==wanted,name
        if (collected/name).exists():assert pin(collected/name)==wanted,name
    for name,link in stage['links'].items():assert payload['files'][name]==link['pin']==pin(collected/name)
    state=read(collected/'identity.json')
    assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS and state['boot_time']==1789634288.0
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    module=importlib.util.spec_from_file_location('timing_accountant',collected/'tools/campaign_processes.py')
    account=importlib.util.module_from_spec(module);module.loader.exec_module(account)
    resources=[]
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']==row['preflight_observations'][-1]
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            assert sample['job']==row['name'];check_sample(sample)
            assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resource=dict(name=row['name'],samples=len(samples),seconds=row['seconds'],peak_rss=row['peak_rss'])
        if row['name'] in TIMING_JOBS:
            foreign=account.foreign_fraction(read(collected/row['name']/'pre.json'),read(collected/row['name']/'post.json'),state['supervisor']['pid'])
            assert foreign==row['accounting'] and foreign['valid'] and foreign['foreign_cpu_fraction']<=.01
            resource['accounting']=foreign
        resources.append(resource)
    assert state['ended']-state['started']<4*3600 and (collected/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    built=read(collected/'built.json');assert built['passed'] and built['identities']==payload['identities']
    for name,wanted in built['files'].items():assert pin(collected/name)==wanted,name
    for role,identities in payload['identities'].items():
        for name,wanted in identities.items():assert pin(collected/'runtimes'/role/name)==wanted==pin(collected/'products'/role/name)
        assert pin(collected/'runtimes'/role/'ParakeetRecurrenceTiming.dll')==built['consumer']
    reviews=[];workers={}
    for worker in state['runs'][3:]:
        name=worker['name'];review=qualify(collected,name,payload,built,worker)
        assert review==read(collected/name/'review.json');reviews.append(review);workers[name]=read(collected/name/'output/result.json')
    performance=evaluate(workers,list(stage['cases']))
    analysis=dict(passed=True,identities=payload['identities'],consumer=built['consumer'],reviews=reviews,resources=resources,performance=performance,
        complete_call_clocks=30400,warmup=15200,measured=15200,exact_output_arrays=91200,
        source_prepared=pin(collected/'evidence/source-prepared.json'),root_product_changed=False)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,admitted=performance['admitted'],files=files,local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),admitted=performance['admitted'],controls_passed=performance['controls_passed'],
        failed_controls=[r for r in performance['controls'] if not r['passed']],failed_gates=[r for r in performance['gates'] if not r['passed']],table=performance['table'])))


if __name__=='__main__':main()
