"""Independently check every retained call, exact outputs and the unchanged gates."""
import csv,json
import numpy as np
from prepare import ROOT,BASE,GRAPH,DIAGNOSTIC,previous_closed
from protocol import pin,read,save,JOBS,BUILD_JOBS,CASES,ORDER,LIMITS,check_sample
from checks import check_result,consumer_inventory
from statistics import summarize


def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    from run import prepared
    prepared();c=BASE/'collected';receipt=read(c/'collection.json');state=read(c/'identity.json');spec=read(BASE/'payload.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    transfer=read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    assert receipt['payload']==pin(BASE/'payload.json')
    assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json') and state['boot_time']==1789634288.0
    assert [r['name'] for r in state['runs']]==JOBS and state['ended']-state['started']<4*3600
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    assert spec['products']==read(GRAPH/'payload.json')['products']==read(DIAGNOSTIC/'payload.json')['products']
    original=read(GRAPH/'collected/cases.json');case,=[row for row in original['cases'] if row['key']=='e5-8tok']
    assert read(c/'cases.json')==dict(original,cases=[case])
    for role in ['current','candidate']:
        manifest=read(c/f'cases-{role}.json')
        assert manifest==dict(original,cases=[case],core=spec['products'][role]['Lokad.Onnx.dll']['sha256'])
    for name,wanted in spec['files'].items():
        if name.startswith(('reference/','runtimes/','previous/','product/','source/','bridge/')):assert pin(c/name)==wanted,name
    built=read(c/'built.json');assert built['passed']
    for name,wanted in built['files'].items():assert pin(c/name)==wanted,name
    consumer=consumer_inventory(read(c/'consumer-inventory/instructions.json'),spec,built)
    assert consumer==read(c/'consumer-inventory/review.json')
    assert (c/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    reports={};resources=[];clocks=[];setups=[]
    for row in state['runs']:
        name=row['name'];assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (c/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:assert row['members'][str(member['pid'])]==member['birth']
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources.append(dict(name=name,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
        if name in BUILD_JOBS:continue
        value=check_result(c,name,spec,row);reports[name]=value
        clocks.extend(dict(process=name,**clock) for clock in value['clocks'])
        setups.append(dict(process=name,seconds=value['setup_seconds']))
    assert len(reports)==9 and CASES==['e5-8tok']
    native=reports['verify-e5-8tok-ort'];native_folder=c/'verify-e5-8tok-ort/output'
    expected_by_role={};numerics=[]
    for name,value in reports.items():
        role=name.split('-')[-1] if name.startswith('verify-') else name.split('-')[-2]
        hashes=[a['sha256'] for a in value['arrays']]
        assert expected_by_role.setdefault(role,hashes)==hashes,(name,'same-role output drift')
        old=read(GRAPH/'collected'/f'verify-e5-8tok-{role}'/'output/result.json')
        assert [(a['name'],a['shape'],a['sha256']) for a in value['arrays']]==[(a['name'],a['shape'],a['sha256']) for a in old['arrays']]
        for actual,reference in zip(value['arrays'],native['arrays'],strict=True):
            assert (actual['name'],actual['shape'])==(reference['name'],reference['shape'])
            a=np.fromfile(c/name/'output'/actual['file'],dtype='<f4').astype(np.float64)
            b=np.fromfile(native_folder/reference['file'],dtype='<f4').astype(np.float64)
            assert np.isfinite(a).all() and np.isfinite(b).all() and a.size==b.size==actual['values']
            error=float(np.max(np.abs(a-b)/np.maximum(1,np.abs(b))))
            assert error<=1e-4,(name,actual['name'],error)
            numerics.append(dict(process=name,output=actual['name'],values=int(a.size),max_scaled_error=error))
    assert expected_by_role['current']==expected_by_role['candidate']
    performance=dict(key='e5-8tok',**summarize({role:reports['timing-e5-8tok-'+role] for role in ORDER}))
    assert len(clocks)==37089 and sum(not r['warmup'] for r in clocks)==1080 and len(setups)==9
    save(BASE/'analysis.json',dict(passed=True,performance=performance,numerics=numerics,resources=resources,
        consumer=consumer,clocks=len(clocks),measured=1080,setups=setups,products=spec['products'],root_product_changed=False,
        original_graph_closure=pin(GRAPH/'closed.json'),diagnosis_closure=pin(DIAGNOSTIC/'closed.json')))
    with (BASE/'clocks.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(clocks[0]));writer.writeheader();writer.writerows(clocks)
    save(BASE/'closed.json',dict(passed=True,admitted=performance['qualified'],all_controls_passed=all(c['passed'] for c in performance['controls']),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),performance=performance,resources=sum(r['samples'] for r in resources))))


if __name__=='__main__':main()
