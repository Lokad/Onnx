"""Independently audit four complete kernel grids and all preserved attempts."""
from collections import defaultdict
from fractions import Fraction
from common import *

BALANCED=[0,1,11,2,10,3,9,4,8,5,7,6]
ROLES=['public','pack','consume-base','consume-128','consume-256','consume-512','consume-six',
       'combined-base','combined-128','combined-256','combined-512','combined-six']


def main():
    assert not (BASE/'closed.json').exists()
    capture=read(BASE/'capture-closed.json');assert capture['passed'];verify(capture['files'])
    verify(read(BASE/'driver-prepared.json')['files'])
    captured=read(BASE/'capture/result.json');fixtures=captured['entries']
    nodes={n['id']:n for n in read(TRACE/'trace-output/graphs.json')['encoder']}
    assert captured['taps']==list(dict.fromkeys(x for i in [99,102,109] for x in [nodes[i]['inputs'][0],nodes[i]['outputs'][0]]))
    failed=ROOT/'artifacts/parakeet-wide-matmul-v2-20260921'
    assert pin(failed/'failure.json')['sha256']=='5fcf9a6a574b3c441e41b33740d8739ee67b7f0dc636b5045bf44313d84673e4'
    failure=read(failed/'failure.json');assert not failure['passed']
    for name,wanted in failure['files'].items():assert pin(failed/name)==wanted,name
    for identity in failure['identities_terminal']:terminal(identity)
    timings=defaultdict(list);workers=[];identities=[];counts=defaultdict(int)
    for ordinal in range(4):
        name=f'probe-{ordinal}';state=read(BASE/(name+'-state.json'))
        assert state['complete'] and state['passed'] and state['code']==0 and state['ordinal']==ordinal
        for key in ('supervisor','worker'):terminal(state[key]);identities.append(state[key])
        resources=[json.loads(s) for s in (BASE/(name+'-resources.jsonl')).read_text().splitlines()]
        assert len(resources)==state['samples']>0 and max(r['rss'] for r in resources)==state['peak_rss']
        assert state['preflight']['available']>=4*1024**3
        assert all(r['seconds']<600 and r['rss']<2*1024**3 and r['available']>=1024**3 and r['disk']>=20*1024**3 and r['affinity']==[2] and r['bytes']<=64*1024**2 for r in resources)
        assert all(0<=b['seconds']-a['seconds']<10 for a,b in zip(resources,resources[1:]))
        result=read(BASE/name/'result.json');assert result['passed'] and result['ordinal']==ordinal and result['tests']==144
        identity=result['identity']
        assert {k:v for k,v in identity.items() if k!='runner_sha256'}=={k:v for k,v in captured['identity'].items() if k!='runner_sha256'}
        assert identity['runner_sha256']==pin(BASE/'bin/Probe.dll')['sha256']
        case_order=list(range(12)) if ordinal%2==0 else list(reversed(range(12)))
        assert [v['case_index'] for v in result['checks']]==case_order
        for value in result['checks']:
            f=fixtures[value['case_index']]
            assert value['passed'] and [value[k] for k in ['m','k','n']]==[f[k] for k in ['m','k','n']]
            assert value['expected_sha256']==f['y']['sha256']
        rows=[json.loads(s) for s in (BASE/name/'samples.jsonl').read_text().splitlines()]
        assert len(rows)==result['samples'];last=0;observed=[];grouped=defaultdict(list)
        for row in rows:
            assert row['ordinal']==ordinal and row['role'] in ROLES and row['phase'] in ('warmup','conditioning','measured')
            assert 0<=row['case_index']<12 and row['frequency']>0 and last<=row['start_ticks']<row['end_ticks'];last=row['end_ticks']
            grouped[(row['case_index'],row['phase'],row['role'])].append(row)
            counts[row['phase']]+=1
            if row['phase']=='measured':
                observed.append((row['case_index'],row['repetition'],row['role']))
                timings[(ordinal,row['case_index'],row['role'])].append(Fraction(row['end_ticks']-row['start_ticks'],row['frequency']))
        expected=[(ci,repetition,ROLES[(BALANCED[ri if ordinal%2==0 else 11-ri]+ordinal*3+repetition)%len(ROLES)]) for ci in case_order for repetition in range(12) for ri in range(len(ROLES))]
        assert observed==expected
        nonwarm=[(r['case_index'],r['phase'],r['repetition'],r['role']) for r in rows if r['phase']!='warmup']
        expected_nonwarm=[item for ci,rep,role in expected for item in
            [(ci,'conditioning',rep*3+i,role) for i in range(3)]+[(ci,'measured',rep,role)]]
        assert nonwarm==expected_nonwarm
        for ci in case_order:
            for role in ROLES:
                warm=grouped[(ci,'warmup',role)];measured=grouped[(ci,'measured',role)]
                assert len(warm)>=32 and [r['repetition'] for r in warm]==list(range(len(warm)))
                assert len(measured)==12 and max(r['end_ticks'] for r in warm)<=min(r['start_ticks'] for r in measured)
        workers.append(dict(ordinal=ordinal,samples=len(rows),resources=state['samples'],peak_rss=state['peak_rss']))
    assert len({(i['pid'],i['birth']) for i in identities})==8
    observations=[]
    for ci,f in enumerate(fixtures):
        means={(w,role):sum(timings[(w,ci,role)])/12 for w in range(4) for role in ROLES}
        overall={role:sum(means[(w,role)] for w in range(4))/4 for role in ROLES}
        observations.append(dict(case_index=ci,name=f['name'],node=f['node'],m=f['m'],k=f['k'],n=f['n'],
            milliseconds={r:float(t*1000) for r,t in overall.items()},
            combined_ratios={r:float(overall['combined-'+r]/overall['combined-base']) for r in ['128','256','512','six']},
            per_worker=[dict(ordinal=w,milliseconds={r:float(means[(w,r)]*1000) for r in ROLES},
                combined_ratios={r:float(means[(w,'combined-'+r)]/means[(w,'combined-base')]) for r in ['128','256','512','six']}) for w in range(4)]))
    controls=[]
    for row in observations:
        if row['m']>=64:continue
        ms=row['milliseconds'];names=['combined-'+r for r in ['base','128','256','512','six']]
        aggregate=max(ms[r] for r in names)/min(ms[r] for r in names)
        ratios=[max(w['milliseconds'][r] for r in names)/min(w['milliseconds'][r] for r in names) for w in row['per_worker']]
        controls.append(dict(case_index=row['case_index'],aggregate_max_min=aggregate,worker_max_min=ratios,
            passed=aggregate<=1.03 and all(v<=1.10 for v in ratios)))
    valid=all(c['passed'] for c in controls);assert len(controls)==3
    territory=[r for r in observations if r['m']>=64 and r['m']%3!=0];assert len(territory)==6
    candidates={}
    for kind in ['128','256','512','six']:
        ratios=[r['combined_ratios'][kind] for r in territory]
        individual=[w['combined_ratios'][kind] for r in territory for w in r['per_worker']]
        candidates[kind]=dict(mean_cell_ratio=sum(ratios)/len(ratios),max_cell_ratio=max(ratios),max_worker_ratio=max(individual),
            eligible_for_product_trial=valid and sum(ratios)/len(ratios)<=.95 and max(ratios)<=1.01 and max(individual)<=1.05)
    analysis=dict(passed=True,attribution_valid=valid,identical_controls=controls,candidates=candidates,
        workers=workers,counts=dict(counts),observations=observations,
        scope='Four local fresh-process diagnostic grids; means and variation retained; no whole-application or AMD/native performance claim')
    save(BASE/'analysis.json',analysis)
    files=dict(capture['files'])
    for folder in [BASE,TOOLS]:
        for p in folder.rglob('*'):
            if p.is_file() and not {'packages','obj'}.intersection(p.relative_to(folder).parts) and p.name not in ('finish-state.json','audit-supervisor.log'):
                files[p.relative_to(ROOT).as_posix()]=pin(p)
    files[(failed/'failure.json').relative_to(ROOT).as_posix()]=pin(failed/'failure.json')
    save(BASE/'closed.json',dict(passed=True,files=files,identities=identities,analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(passed=True,attribution_valid=valid,controls=controls,candidates=candidates,counts=dict(counts),workers=workers,closed=pin(BASE/'closed.json'))))
    for row in observations:print(json.dumps({k:v for k,v in row.items() if k!='per_worker'}))


if __name__=='__main__':main()
