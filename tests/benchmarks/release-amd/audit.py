"""Independently verify collection, every clock, fresh native outputs and controls."""
import csv,json
import numpy as np
from prepare import ROOT,BASE,previous_closed
from protocol import pin,read,save,JOBS,CASES,ORDER,LIMITS,check_sample
from checks import check_result
from statistics import summarize

def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    c=BASE/'collected';receipt=read(c/'collection.json');state=read(c/'identity.json');spec=read(BASE/'payload.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    transfer=read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    assert receipt['payload']==pin(BASE/'payload.json')
    assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json') and state['boot_time']==1789634288.0
    assert [r['name'] for r in state['runs']]==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    for name,wanted in read(c/'built.json')['files'].items():assert pin(c/name)==wanted,name
    reports={};resources=[];clocks=[]
    for row in state['runs']:
        name=row['name'];assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']['available']>=LIMITS['build_preflight_available' if name in JOBS[:3] else 'preflight_available']
        assert row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (c/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for m in sample['members']:assert row['members'][str(m['pid'])]==m['birth']
        resources.append(dict(name=name,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
        if name in JOBS[:3]:continue
        v=check_result(c,name,spec,row);reports[name]=v
        clocks.extend(dict(process=name,**clock) for clock in v['clocks'])
    performance=[];numerics=[]
    for key in CASES:
        cases={name:v for name,v in reports.items() if v['key']==key};assert len(cases)==6
        native=reports['verify-'+key+'-ort'];native_folder=c/('verify-'+key+'-ort')/'output'
        expected_by_role={}
        for name,v in cases.items():
            hashes=[a['sha256'] for a in v['arrays']]
            assert expected_by_role.setdefault(v['role'],hashes)==hashes,(key,name,'same-role output drift')
            for actual,reference in zip(v['arrays'],native['arrays'],strict=True):
                assert (actual['name'],actual['shape'])==(reference['name'],reference['shape'])
                a=np.fromfile(c/name/'output'/actual['file'],dtype='<f4').astype(np.float64)
                b=np.fromfile(native_folder/reference['file'],dtype='<f4').astype(np.float64)
                assert np.isfinite(a).all() and np.isfinite(b).all() and a.size==b.size==actual['values']
                error=float(np.max(np.abs(a-b)/np.maximum(1,np.abs(b))))
                assert error<=1e-4,(name,actual['name'],error)
                numerics.append(dict(process=name,output=actual['name'],values=int(a.size),max_scaled_error=error))
        performance.append(dict(key=key,**summarize({role:reports['timing-'+key+'-'+role] for role in ORDER})))
    assert len(clocks)==3888 and sum(not r['warmup'] for r in clocks)==1920
    save(BASE/'analysis.json',dict(passed=True,performance=performance,numerics=numerics,resources=resources,clocks=len(clocks),measured=1920,root_product_changed=False))
    with (BASE/'clocks.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(clocks[0]));w.writeheader();w.writerows(clocks)
    save(BASE/'closed.json',dict(passed=True,all_controls_passed=all(r['qualified'] for r in performance),files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),performance=performance,resources=sum(r['samples'] for r in resources))))

if __name__=='__main__':main()
