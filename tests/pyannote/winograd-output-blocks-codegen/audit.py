import json
from prepare import ROOT,BASE,previous_closed
from protocol import JOBS,LIMITS,pin,read,save,check_sample
from numerical_checks import result,same_current,identity
from listings import listings

def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    spec=read(BASE/'prepared.json')
    for name,wanted in spec['files'].items():assert pin(ROOT/name)==wanted,name
    c=BASE/'collected';receipt=read(c/'collection.json');state=read(c/'identity.json')
    payload=read(BASE/'payload.json');transfer=read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    assert spec['archive']==pin(BASE/'payload.tar.gz') and spec['stage']==pin(BASE/'bundle/stage.json')
    built=read(c/'built.json');assert built['passed']
    for name,wanted in built['files'].items():assert pin(c/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items():assert pin(c/'runtimes'/role/name)==wanted,name
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0
    assert state['supervisor']==read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS
    assert state['ended']-state['started']<4*3600
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    fixtures={(r['case'],r['index']):r for r in read(BASE/'bundle/evidence/fixtures.json')['calls'] if r['eligible'] and r['attributes']['strides']==[1,1]}
    rows=[];code={};reports={}
    for row in state['runs']:
        name=row['name'];role,mode,width_text=name.split('-');width=int(width_text);assert mode=='captured'
        assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(line) for line in (c/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for m in sample['members']:assert row['members'][str(m['pid'])]==m['birth']
        value=read(c/name/'result.json');reference=read(BASE/'bundle/evidence'/(name+'.json'))
        identity(value,role,payload,built,row)
        assert value['assembly']==reference['assembly'] and value['core_sha256']==reference['core_sha256']
        same_current(value,reference)
        assert value['rows']==reference['rows'] and value['contracts']==value['refusals']==0
        reports[name]=result(value,'captured',width,fixtures);assert not reports[name]['failures']
        code[name]=listings(c/name/'jit.asm');assert code[name]
        assert all(b['complete_body'] and b['complete_uninterleaved'] and not b['managed_stdout_repairs'] for b in code[name])
        assert any(':TransformWinogradInputContiguous(' in b['method'] and (b['tier'].startswith('Tier1') or b['tier']=='Tier0-FullOpts') for b in code[name]),name
        assert value['rangeChecks']==value['singleBlockRefusals']==0
        assert any(':EpilogueRange(' in b['method'] and b['tier'].startswith('Tier1') for b in code[name]),name
        for method in ['MultiplyWinograd','OutputWinograd']:
            assert any((':'+method+str(width)+'(') in b['method'] and b['tier'].startswith('Tier1') for b in code[name]),(name,method)
        rows.append(dict(name=name,samples=len(samples),peak_rss=row['peak_rss'],bodies=len(code[name])))
    save(BASE/'listings.json',code)
    save(BASE/'analysis.json',dict(passed=True,products=payload['products'],consumer=built['consumer'],resources=rows,numerical_results=reports,
        all_tiers_retained=True,no_performance_measurement=True,manual_review_pending=True))
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,manual_review_pending=True,no_performance_measurement=True))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),resources=rows,methods={name:[dict(method=b['method'].split('(')[0],tier=b['tier'],bytes=b['code_bytes']) for b in bodies] for name,bodies in code.items()})))

if __name__=='__main__':main()
