"""Independently check complete banks, every observation and fixed performance gates."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,statistics
import numpy as np
from common import CORE,LIMITS,BANKS,CASES,VARIANTS,order,pin,read,write,verify
from data import inputs,describe,scalar,digest

def inspect_timing(value,visit,probe,descriptions):
    assert value['schema']==1 and value['protocol']=='layernorm-complete-bank-v1' and value['visit']==visit
    identity=value['identity'];assert identity['mode']=='run' and identity['core_sha256']==CORE and identity['probe_sha256']==probe['sha256']
    assert identity['runtime']=='10.0.8' and identity['affinity']==4 and identity['vector_width']==8 and identity['settings']=={}
    assert identity['vector512_hardware'] is True and identity['avx512'] is True and value['frequency']==1000000000
    assert value['case_order']==order(visit) and [c['name'] for c in value['results']]==[d['name'] for d in BANKS]
    rows=[]
    for definition,case,description in zip(BANKS,value['results'],descriptions):
        assert case['nodes']==25 and case['values']==25*definition['rows']*definition['width'] and case['held_unchanged'] is True
        for key,want in description.items():
            if want is not None:assert case[key]==want,(case['name'],key)
        assert len(case['output_sha256'])==64
        for label,cycles in [('first',1),('warmup',16),('measured',48)]:
            samples=case[label];assert len(samples)==cycles*4
            for index,sample in enumerate(samples):
                cycle,position=divmod(index,4);variant=position if label=='first' else (visit+cycle+position)%4
                assert (sample['cycle'],sample['position'],sample['variant'],sample['repeats'])==(-1 if label=='first' else cycle,position,VARIANTS[variant],1 if label=='first' else definition['repeats'])
                assert type(sample['ticks']) is int and sample['ticks']>0 and type(sample['allocated']) is int and sample['allocated']>=0
                assert len(sample['gc'])==3 and all(type(v)is int and v>=0 for v in sample['gc'])
                assert sample['output_sha256']==case['output_sha256']
        for variant in VARIANTS:
            samples=[s for s in case['measured'] if s['variant']==variant];assert len(samples)==48
            times=[s['ticks']/s['repeats']/value['frequency']*1000 for s in samples]
            rows.append(dict(case=case['name'],visit=visit,variant=variant,mean_ms=statistics.mean(times),median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
                ticks=sum(s['ticks'] for s in samples),banks=sum(s['repeats'] for s in samples),allocated=sum(s['allocated'] for s in samples),gc=[sum(s['gc'][i] for s in samples) for i in range(3)]))
    return rows

def summarize(rows):
    assert len(rows)==144 and len({(r['case'],r['visit'],r['variant']) for r in rows})==144
    cases=[]
    for definition in BANKS:
        name=definition['name'];means={v:statistics.mean(r['mean_ms'] for r in rows if r['case']==name and r['variant']==v) for v in VARIANTS}
        visits=[]
        for visit in range(4):
            m={r['variant']:r['mean_ms'] for r in rows if r['case']==name and r['visit']==visit};assert set(m)==set(VARIANTS)
            visits.append(dict(visit=visit,duplicate_ratio=m['CopyB']/m['CopyA'],candidate_ratios={v:m['Wide']/m[v] for v in VARIANTS[:3]}))
        duplicate=means['CopyB']/means['CopyA'];ratios={v:means['Wide']/means[v] for v in VARIANTS[:3]}
        control=1/1.01<=duplicate<=1.01 and all(1/1.02<=v['duplicate_ratio']<=1.02 for v in visits)
        regression=all(r<=1.01 for r in ratios.values()) and all(r<=1.02 for v in visits for r in v['candidate_ratios'].values())
        gain=name not in CASES[1:4] or all(r<=.98 for r in ratios.values())
        cases.append(dict(name=name,diagnostic=definition['diagnostic'],means_ms=means,duplicate_ratio=duplicate,candidate_ratios=ratios,visits=visits,controls_passed=control,no_regression_passed=regression,gain_passed=gain))
    controls=all(c['controls_passed'] for c in cases);candidate=all(c['no_regression_passed'] and c['gain_passed'] for c in cases)
    return dict(cases=cases,controls_passed=controls,candidate_screen_passed=candidate,overall_passed=controls and candidate,
        verdict='PASS: complete-kernel screen only' if controls and candidate else 'INCONCLUSIVE: duplicate controls fail' if not controls else 'REJECT: candidate misses fixed screen')

def resources(run,samples):
    assert run['code']==0 and run['terminal_members'] is True and 0<run['seconds']<LIMITS['seconds'] and run['started']<run['ended']
    assert len(samples)==run['samples']>0 and samples[0]['seconds']<5 and run['seconds']-samples[-1]['seconds']<5
    assert all(0<b['seconds']-a['seconds']<5 for a,b in zip(samples,samples[1:]))
    observed={};last_cpu={};peak=0
    for sample in samples:
        assert 0<=sample['seconds']<=run['seconds'] and sample['available']>=LIMITS['available']
        assert len({p['pid'] for p in sample['members']})==len(sample['members'])
        for p in sample['members']:
            key=str(p['pid']);assert p['affinity']==[2] and p['rss']>=0 and p['birth']>=run['child']['birth'] and p['birth']==run['members'][key]
            assert key not in observed or observed[key]==p['birth'];observed[key]=p['birth']
            assert math.isfinite(p['cpu']) and p['cpu']>=last_cpu.get(key,0);last_cpu[key]=p['cpu']
        peak=max(peak,sum(p['rss'] for p in sample['members']))
    assert observed==run['members'] and observed[str(run['child']['pid'])]==run['child']['birth'] and peak==run['peak_rss']<=LIMITS['rss']
    return dict(samples=len(samples),peak_rss=peak,minimum_available=min(s['available'] for s in samples),last_sampled_cpu=last_cpu,births=[dict(pid=int(pid),birth=birth) for pid,birth in observed.items()])

def audit(base,origin):
    bundle=verify(base,origin);state=read(base/'result/identity.json');probe=pin(base/'bin/LayerNormBank.dll')
    assert state['complete'] is True and state['code']==0 and not state.get('error') and state['limits']==LIMITS and state['bundle']==pin(base/'bundle.json')
    assert [r['visit'] for r in state['runs']]==list(range(4)) and all(a['ended']<=b['started'] for a,b in zip(state['runs'],state['runs'][1:]))
    assert 'AuthenticAMD' in (base/'result/cpuinfo.txt').read_text() and 'avx512f' in (base/'result/cpuinfo.txt').read_text()
    spec=importlib.util.spec_from_file_location('accounting',base/'tools/campaign_processes.py');accounting=importlib.util.module_from_spec(spec);spec.loader.exec_module(accounting)
    descriptions=[describe(origin,d) for d in BANKS];references={d['name']:scalar(inputs(origin,d)) for d in BANKS if d['diagnostic']}
    rows=[];telemetry=[];births=[state['supervisor']];diagnostics=[];hashes=None
    for run in state['runs']:
        folder=base/'result'/str(run['visit']);value=read(folder/'output/timing.json');rows+=inspect_timing(value,run['visit'],probe,descriptions)
        current=[c['output_sha256'] for c in value['results']]
        if hashes is None:hashes=current
        assert hashes==current,'Product/candidate outputs changed between fresh workers'
        for d,c in zip(BANKS,value['results']):
            if not d['diagnostic']:continue
            path=folder/'output'/(d['name']+'-reference.f32');actual=np.fromfile(path,dtype='<f4');reference=references[d['name']]
            assert actual.size==reference.size and np.isfinite(actual).all() and pin(path)['sha256']==c['output_sha256']
            error=float(np.max(np.abs(actual.astype(np.float64)-reference.astype(np.float64))/np.maximum(1,np.abs(reference.astype(np.float64)))))
            assert error<=1e-5;diagnostics.append(dict(visit=run['visit'],name=d['name'],values=int(actual.size),maximum_scalar_error=error))
        r=resources(run,[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]);births+=r['births']
        foreign=accounting.foreign_fraction(read(folder/'pre.json'),read(folder/'post.json'),state['supervisor']['pid']);assert foreign==run['accounting'] and foreign['foreign_cpu_fraction']<=LIMITS['foreign']
        before=[int(x) for x in (folder/'cpu-before.txt').read_text().splitlines()[0].split()[1:9]];after=[int(x) for x in (folder/'cpu-after.txt').read_text().splitlines()[0].split()[1:9]]
        delta=[b-a for a,b in zip(before,after)];assert all(v>=0 for v in delta) and sum(delta)>0
        steal=delta[7]/sum(delta);assert steal<=LIMITS['steal'];telemetry.append(dict(visit=run['visit'],seconds=run['seconds'],**r,foreign=foreign,steal_fraction=steal))
    return dict(execution_passed=True,bundle=pin(base/'bundle.json'),measured_samples=6912,warmup_samples=2304,first_calls=144,rows=rows,resources=telemetry,
        diagnostic_arrays=diagnostics,births=births,**summarize(rows),scope='Complete LayerNorm banks, no full-model/native latency or default qualification')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--payload',type=Path,required=True);parser.add_argument('--origin',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.payload.resolve(),args.origin.resolve());write(args.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','resources','births','diagnostic_arrays']},indent=2))
