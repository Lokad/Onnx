from common import *
import statistics

def main():
    assert not (BASE/'closed.json').exists()
    spec=read(BASE/'prepared.json');verify(spec['files'])
    identities=[];resources=[]
    for filename,expected in [('preparation.json',['restore','build']),('processes.json',spec['jobs'])]:
        st=read(BASE/filename)
        assert st['complete'] and st['code']==0 and [r['name'] for r in st['runs']]==expected
        identities.append(st['supervisor'])
        for run in st['runs']:
            assert run['complete'] and run['code']==0 and run['seconds']<900
            assert run['preflight']['available']>=8*1024**3 and run['preflight']==run['preflight_observations'][-1]
            assert all(r['seconds']<900 and r['disk']>=20*1024**3 for r in run['preflight_observations'])
            samples=[json.loads(line) for line in (BASE/'logs'/(run['name']+'.samples.jsonl')).read_text().splitlines()]
            assert len(samples)==run['samples']>0 and max(s['rss'] for s in samples)==run['peak_rss']
            for s in samples:
                assert s['seconds']<900 and s['rss']<4*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3 and s['output_bytes']<=1024**3
                assert s['rss']==sum(p['rss'] for p in s['members'])
                for p in s['members']: assert run['members'][str(p['pid'])]==p['birth'] and p['affinity']==[2]
            identities.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
            resources.append(dict(name=run['name'],samples=len(samples),peak_rss=run['peak_rss'],seconds=run['seconds']))
    for identity in identities:terminal(identity)
    shapes=read(BASE/'shapes.json')['shapes'];results={}
    for name in spec['jobs']:
        r=read(BASE/'output'/(name+'.json'));results[name]=r
        assert r['passed'] and r['runtime']=='10.0.12' and r['flags']==[] and r['processor_count']==1 and r['fma']
        assert r['core']==CORE and r['executable']==spec['executable']['sha256'] and r['shapes']==pin(BASE/'shapes.json')['sha256']
        assert r['mode']==name.split('-')[0]
        job=next(row for row in read(BASE/'processes.json')['runs'] if row['name']==name)
        assert r['pid']==job['worker']['pid']
    validation=results['validate']['records'];expected=[]
    for m in [0,2,4,6,8,10,16,32,64]:
        for n in [0,1,9,31,64]:
            for k in [0,1,7,8,15,31,32,33,63,64,65]:
                for nonzero in [False,True]:expected.append((m,n,k,nonzero))
    for s in shapes:
        for nonzero in [False,True]:expected.append((s['m'],s['n'],s['k'],nonzero))
    assert [(r['m'],r['n'],r['k'],r['nonzero']) for r in validation]==expected
    assert all(r['passed'] and r['values']==r['m']*r['k'] for r in validation)
    digests={(r['m'],r['n'],r['k']):r['digest'] for r in validation if not r['nonzero']}
    rows=[];control_ok=True
    for s in shapes:
        key=(s['m'],s['n'],s['k']);means={}
        for name in spec['jobs'][1:]:
            result=results[name]
            assert len(result['records'])==6*len(shapes)
            subset=[r for r in result['records'] if (r['m'],r['n'],r['k'])==key]
            assert [r['block'] for r in subset]==list(range(6))
            for r in subset:
                assert r['iterations']==s['iterations'] and r['warm_calls']>=16 and r['warm_seconds']>=1
                assert r['digest']==digests[key] and r['seconds']>0 and 0<=r['cpu_seconds']<=r['seconds']+.1
            means[name]=statistics.fmean(r['seconds']/r['iterations'] for r in subset)
        controls={role:max(means[role+'-a'],means[role+'-b'])/min(means[role+'-a'],means[role+'-b']) for role in ['baseline','candidate']}
        passed=all(v<=spec['gates']['process_max_min'] for v in controls.values());control_ok &= passed
        ratio=(means['candidate-a']+means['candidate-b'])/(means['baseline-a']+means['baseline-b'])
        rows.append(dict(**s,process_means=means,controls=controls,control_passed=passed,candidate_baseline=ratio))
    geomean=math.exp(statistics.fmean(math.log(r['candidate_baseline']) for r in rows))
    worst=max(r['candidate_baseline'] for r in rows)
    eligible=control_ok and geomean<=spec['gates']['geomean_candidate_baseline'] and worst<=spec['gates']['max_shape_candidate_baseline']
    analysis=dict(passed=True,validation_cases=len(validation),validation_values=sum(r['values'] for r in validation),
        controls_passed=control_ok,geomean_candidate_baseline=geomean,max_shape_candidate_baseline=worst,eligible=eligible,
        rows=rows,resources=resources,resource_samples=sum(r['samples'] for r in resources),identities=identities,
        scope=spec['scope'],gates=spec['gates'])
    save(BASE/'analysis.json',analysis)
    files=dict(spec['files'])
    for p in BASE.rglob('*'):
        if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts):files[rel(p)]=pin(p)
    save(BASE/'closed.json',dict(passed=True,files=files,analysis=pin(BASE/'analysis.json'),identities=identities))
    print(dict(eligible=eligible,controls=control_ok,geomean=geomean,worst=worst,validation_cases=len(validation),resources=analysis['resource_samples'],closed=pin(BASE/'closed.json')))
    for r in rows:print((r['m'],r['n'],r['k']),r['candidate_baseline'],r['controls'])

if __name__=='__main__':main()
