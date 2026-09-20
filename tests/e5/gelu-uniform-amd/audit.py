"""Check all fixed timing records and the unchanged prospective engineering screen."""
from pathlib import Path
import argparse,json,math,statistics
from prepare import CASES,pin,read,write

VARIANTS=['Product','CopyA','CopyB','Conditional'];REPEATS=[64,32,8,8,2]

def inspect_timing(value,visit):
    assert value['schema']==1 and value['protocol']=='uniform-gelu-bank-v1' and value['visit']==visit
    assert value['runtime']=='.NET 10.0.8' and value['affinity']==4 and value['avx512'] and value['flags']==[]
    assert value['frequency']==1000000000
    order=[CASES[(i+visit)%5] for i in range(5)]
    if visit%2:order.reverse()
    assert value['case_order']==order and [r['name'] for r in value['results']]==CASES
    rows=[]
    for ci,case in enumerate(value['results']):
        assert case['layers']==12 and case['held_unchanged'] and case['values']==[147456,552960,2359296,2359296,9437184][ci]
        for label,cycles in [('warmup',16),('measured',48)]:
            samples=case[label];assert len(samples)==cycles*4
            for index,s in enumerate(samples):
                cycle,position=divmod(index,4)
                assert (s['cycle'],s['position'],s['variant'],s['repeats'])==(cycle,position,VARIANTS[(visit+cycle+position)%4],REPEATS[ci])
                assert type(s['ticks']) is int and s['ticks']>0 and type(s['allocated']) is int and s['allocated']>=0
                assert len(s['gc'])==3 and all(type(n)is int and n>=0 for n in s['gc'])
                assert s['output_sha256']==case['output_sha256']
        for variant in VARIANTS:
            samples=[s for s in case['measured'] if s['variant']==variant];assert len(samples)==48
            times=[s['ticks']/s['repeats']/value['frequency']*1000 for s in samples]
            rows.append(dict(case=case['name'],visit=visit,variant=variant,mean_ms=statistics.mean(times),median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),allocated=sum(s['allocated'] for s in samples),gc=[sum(s['gc'][i] for s in samples) for i in range(3)]))
    return rows

def summarize(rows):
    assert len(rows)==80;cases=[]
    for name in CASES:
        means={v:statistics.mean(r['mean_ms'] for r in rows if r['case']==name and r['variant']==v) for v in VARIANTS}
        visits=[]
        for visit in range(4):
            m={r['variant']:r['mean_ms'] for r in rows if r['case']==name and r['visit']==visit};assert set(m)==set(VARIANTS)
            visits.append(dict(visit=visit,copy_ratio=m['CopyB']/m['CopyA'],candidate_ratios={v:m['Conditional']/m[v] for v in VARIANTS[:3]}))
        duplicate_ratio=means['CopyB']/means['CopyA'];ratios={v:means['Conditional']/means[v] for v in VARIANTS[:3]}
        control_pass=1/1.01<=duplicate_ratio<=1.01 and all(1/1.02<=v['copy_ratio']<=1.02 for v in visits)
        no_regression=all(r<=1.01 for r in ratios.values()) and all(r<=1.02 for v in visits for r in v['candidate_ratios'].values())
        gain_pass=name not in CASES[1:4] or all(r<=.98 for r in ratios.values())
        cases.append(dict(name=name,means_ms=means,duplicate_ratio=duplicate_ratio,candidate_ratios=ratios,visits=visits,controls_passed=control_pass,no_regression_passed=no_regression,gain_passed=gain_pass))
    controls=all(c['controls_passed'] for c in cases);candidate=all(c['no_regression_passed'] and c['gain_passed'] for c in cases)
    return dict(cases=cases,controls_passed=controls,candidate_screen_passed=candidate,overall_passed=controls and candidate,verdict='PASS: kernel screen only' if controls and candidate else 'INCONCLUSIVE: duplicate controls fail' if not controls else 'REJECT: candidate misses the fixed kernel screen')

def audit(base,payload):
    bundle=read(base/'bundle.json');identity=read(base/'result/identity.json');code=read(base/'code/identity.json')
    assert identity['complete'] and code['complete'] and not identity.get('error') and not code.get('error')
    assert [r['name'] for r in identity['runs']]==['0','1','2','3'] and [r['name'] for r in code['runs']]==['proof']
    assert identity['bundle_sha256']==code['bundle_sha256']==pin(base/'bundle.json')['sha256']
    for name,wanted in bundle['files'].items():assert pin(payload/name)==wanted,name
    rows=[];resources=[];hashes=None
    for phase in ['code','result']:
        state=code if phase=='code' else identity
        assert state['limits']==dict(rss=3*1024**3,seconds=600,available_memory=1024**3)
        for r in state['runs']:
            assert r['code']==0 and 0<r['seconds']<600
            proof=read(base/phase/(r['name']+'.json'))
            assert proof['passed'] and proof['cases']==1575 and proof['compared']==102364884 and proof['avx512'] and proof['runtime']=='.NET 10.0.8'
            assert proof['probe_sha256']==bundle['files']['bin/Probe.dll']['sha256'] and proof['core_sha256']==bundle['files']['bin/Lokad.Onnx.dll']['sha256']
            assert len(proof['captures'])==60
            for c in proof['captures']:
                for kind,key in [('x','input_sha256'),('bias','bias_sha256')]:assert c[key]==bundle['files'][f"data/capture/{c['name']}/{c['layer']:02d}-{kind}.f32"]['sha256']
            samples=[json.loads(line) for line in (base/phase/(r['name']+'-samples.jsonl')).read_text().splitlines()]
            assert len(samples)==r['samples'] and samples
            peak=max(sum(m['rss'] for m in s['members']) for s in samples);assert peak==r['peak_rss']
            for s in samples:
                assert 0<=s['seconds']<600 and s['available_memory']>=1024**3 and sum(m['rss'] for m in s['members'])<3*1024**3
                for m in s['members']:assert m['affinity']=='2' and m['group']==r['pid'] and m['start']>=r['start']
            resources.append(dict(phase=phase,name=r['name'],seconds=r['seconds'],peak_rss=peak,minimum_available=min(s['available_memory'] for s in samples),samples=len(samples),accounting=r['accounting']))
            if phase=='result':
                value=read(base/phase/(r['name']+'.json.timing.json'));rows+=inspect_timing(value,int(r['name']))
                current=[(c['input_sha256'],c['bias_sha256'],c['output_sha256']) for c in value['results']]
                for c in value['results']:
                    assert {k:c[k] for k in ('input_sha256','bias_sha256','output_sha256')}==bundle['banks'][c['name']]
                if hashes is None:hashes=current
                assert hashes==current
                assert r['flags']=={}
    gate=read(base/'code-gate.json');assert gate['passed'] and gate['jit_sha256']==pin(base/'code/jit.txt')['sha256'] and gate['proof_sha256']==pin(base/'code/proof.json')['sha256'] and gate['probe_sha256']==bundle['files']['bin/Probe.dll']['sha256']
    return dict(execution_passed=True,measured_batches=3840,warmup_batches=1280,bank_calls=87552,rows=rows,resources=resources,**summarize(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=audit(a.artifact,a.payload);write(a.output,value);print(json.dumps({k:v for k,v in value.items() if k not in ['rows','resources']},indent=2))
