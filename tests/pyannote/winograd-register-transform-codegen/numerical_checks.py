"""Audit the complete fixed census and all resource/identity records."""
import collections
import itertools
import json
import math
from pathlib import Path
from protocol import JOBS,LIMITS,pin,read,save,check_sample
from prepare import ROOT,BASE,previous_closed

PAIRS=[(16,32),(16,48),(32,32),(32,64),(64,48),(64,64),(128,128),(256,256)]
SHAPES=[(1,1),(1,7),(3,5),(4,8),(5,17),(6,16)]
WIDE=[(5,33),(6,34),(5,65),(6,66)]

def raw_census():
    original=[row for shapes in [SHAPES,WIDE] for row in itertools.product(PAIRS,shapes,['random','impulse','cancellation'],range(8))]
    extra=list(itertools.product(list(itertools.product([16,32],[32,48,64,80,96,128,256,512])),
        [(1,1),(3,17),(5,33),(6,34)],['random','impulse','cancellation'],range(8)))
    assert len(original)==1920 and len(extra)==1536
    return original+extra


def error(value,count):
    assert value['Count']==count and 0<=value['WorstIndex']<count
    assert 0<=value['DifferentBits']<=count
    assert all(math.isfinite(value[k]) for k in ['Absolute','Scaled','Actual','Reference'])
    assert value['Absolute']>=0 and value['Scaled']>=0
    worst=abs(value['Actual']-value['Reference'])/max(1,abs(value['Reference']))
    assert math.isclose(worst,value['Scaled'],rel_tol=1e-12,abs_tol=1e-15)
    assert value['Passed']==(value['Scaled']<=1e-4)
    return value['Passed']


def result(value,mode,width,fixtures):
    assert value['completed'] and value['noPerformanceMeasurement']
    assert value['mode']==mode and value['width']==width and value['runtime']=='10.0.8'
    rows=value['rows'];assert len(rows)==(3456 if mode=='raw' else 87)
    assert value['refusals']==(21 if mode=='raw' else 0)
    assert value['contracts']==(24 if mode=='raw' else 0)
    assert value['rangeChecks']==(10566 if mode=='raw' else 0)
    assert value['singleBlockRefusals']==(2 if mode=='raw' else 0)
    failures=[];seen=set();outputs={};values=0
    raw_expected=raw_census()
    for index,row in enumerate(rows):
        if mode=='raw':
            (c,m),(h,w),pattern,epilogue=raw_expected[index]
            assert [row[k] for k in ['index','c','m','h','w','pattern','epilogue']]==[index,c,m,h,w,pattern,epilogue]
            key=str(index)
            if pattern=='impulse':assert row['candidate']['Absolute']==0
        else:
            key=(row['fixture'],row['index']);assert key in fixtures and key not in seen;seen.add(key)
            original=fixtures[key];n,c,h,w=original['input']['shape'];m=original['weights']['shape'][0]
            assert [row[k] for k in ['c','m','h','w','form','relu']]==[c,m,h,w,original['form'],original['relu']]
            assert row['native']==original['output']['sha256'] and row['preparedBytes']==16*c*m*4
        count=m*h*w;values+=count
        for label in ['candidate','current','selectedDifference']:
            passed=error(row[label],count)
            if label!='selectedDifference' and not passed:failures.append(dict(case=str(key),role=label,error=row[label]))
        assert all(row[k] is True for k in ['repeated','heldOutput','readonlyInputs','guards'])
        assert len(row['output'])==64 and all(c in '0123456789abcdef' for c in row['output'])
        outputs[str(key)]=row['output']
    if mode=='captured':assert seen==set(fixtures)
    return dict(cases=len(rows),values=values,failures=failures,outputs=outputs,
        maximum_candidate_scaled=max(r['candidate']['Scaled'] for r in rows),
        maximum_current_scaled=max(r['current']['Scaled'] for r in rows),
        candidate_differing_bits=sum(r['candidate']['DifferentBits'] for r in rows))


def same_previous(actual,previous):
    for key in ['mode','width','runtime','refusals','contracts','rangeChecks','completed','noPerformanceMeasurement']:
        assert actual[key]==previous[key],('M33 comparison',key)
    count=1920 if actual['mode']=='raw' else 87
    assert len(previous['rows'])==count
    assert actual['rows'][:count]==previous['rows'],('M33 comparison','rows')
    return True


def same_current(actual,current):
    for key in ['mode','width','runtime','refusals','contracts','rangeChecks','singleBlockRefusals',
                'completed','noPerformanceMeasurement','rows']:
        assert actual[key]==current[key],('fresh product comparison',key)
    return True


def identity(value,role,payload,built,row):
    assert value['assembly']==built['consumer']['sha256'] and value['pid']==row['child']['pid']
    assert value['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']


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
        resources+=len(samples);peak=max(peak,row['peak_rss'])
    built=read(folder/'built.json');assert built['passed']
    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items(): assert pin(folder/'runtimes'/role/name)==wanted
    fixtures={(r['case'],r['index']):r for r in read(BASE/'bundle/evidence/result.json')['calls']
        if r['eligible'] and r['attributes']['strides']==[1,1]}
    assert len(fixtures)==87
    assert collections.Counter(r['form'] for r in fixtures.values())=={1:9,2:9,5:12,6:9,9:18,10:15,13:9,14:6}
    results={};values={}
    for row in state['runs'][3:]:
        name=row['name'];role,mode,width=name.split('-');value=read(folder/name/'result.json');values[name]=value
        identity(value,role,payload,built,row)
        results[name]=result(value,mode,int(width),fixtures)
        same_previous(value,read(BASE/'bundle/evidence'/('previous-'+mode+'-'+width+'.json')))
        results[name]['previous_M33_exact']=True
        same_current(value,read(BASE/'bundle/evidence'/('previous-full-'+mode+'-'+width+'.json')))
        results[name]['previous_full_M36_exact']=True
    for mode in ['raw','captured']:
        for role in ['current','candidate']:
            assert results[role+'-'+mode+'-256']['outputs']==results[role+'-'+mode+'-512']['outputs']
        for width in [256,512]:
            name='candidate-'+mode+'-'+str(width)
            same_current(values[name],values['current-'+mode+'-'+str(width)])
            results[name]['fresh_current_exact']=True
    admitted=not any(v['failures'] for v in results.values())
    analysis=dict(passed=True,numerically_admitted=admitted,no_performance_measurement=True,root_product_changed=False,
        consumer=built['consumer'],products=payload['products'],resources=resources,peak_rss=peak,results=results)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in folder.rglob('*') if p.is_file()}
    for name in ['analysis.json','prepared.json','staged.json','payload.json','deployment.json','collection-transfer.json','results.tar.gz','payload.tar.gz']:
        files[name]=pin(BASE/name)
    save(BASE/'closed.json',dict(passed=True,numerically_admitted=admitted,files=files,root_product_changed=False,no_performance_measurement=True))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),numerically_admitted=admitted,resources=resources,peak_rss=peak,
        results={k:{n:v[n] for n in ['cases','values','maximum_candidate_scaled','maximum_current_scaled','failures']} for k,v in results.items()})))


if __name__=='__main__':main()
