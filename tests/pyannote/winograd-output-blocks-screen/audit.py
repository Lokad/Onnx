import json
from prepare import ROOT,BASE,previous_closed
from protocol import JOBS,LIMITS,pin,read,save,check_sample
from score import fixtures,check_result,score,ORDER

def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    c=BASE/'collected';receipt=read(c/'collection.json');state=read(c/'identity.json');payload=read(BASE/'payload.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    transfer=read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    for name,wanted in read(BASE/'bundle/stage.json')['files'].items():assert payload['files'][name]==pin(BASE/'bundle'/name)==wanted,name
    assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0
    assert state['supervisor']==read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']]==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[];reports={};built=read(c/'built.json');assert built['passed']
    for name,wanted in built['files'].items():assert pin(c/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items():assert pin(c/'runtimes'/role/name)==wanted,name
    spec=read(BASE/'bundle/evidence/fixtures.json');calls=fixtures(spec);reference=read(BASE/'bundle/reference.json')
    for row in state['runs']:
        name=row['name'];assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        limit=LIMITS['build_preflight_available' if name in JOBS[:3] else 'preflight_available']
        assert row['preflight']['available']>=limit and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (c/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for m in sample['members']:assert row['members'][str(m['pid'])]==m['birth']
        resources.append(dict(name=name,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
        if name in ['verify',*ORDER]:
            value=read(c/name/'result.json')
            assert value['pid']==row['child']['pid'] and value['runtime']=='10.0.8' and value['consumer']==built['consumer']['sha256']
            if name=='verify':assert check_result(value,calls,reference)==dict(passed=True,calls=174)
            else:reports[name]=value
    performance=score(reports,spec,reference)
    save(BASE/'analysis.json',dict(passed=True,performance=performance,resources=resources,built=built,
        verification_calls=174,root_product_changed=False,no_application_or_ort_timing=True))
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,admitted=performance['admitted'],files=files,
        root_product_changed=False,no_application_or_ort_timing=True))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),admitted=performance['admitted'],aggregate=performance['rows'][0],
        failed_controls=[r for r in performance['controls'] if not r['passed']],failed_gates=[r for r in performance['gates'] if not r['passed']],
        separation=performance['process_separation'],resources=resources)))

if __name__=='__main__':main()
