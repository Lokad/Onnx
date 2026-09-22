"""Independently check retained observations and the prospective component gates."""
from common import *
from generate import generate
from transport import ssh, PRELUDE
import math
import statistics


def main():
    assert not (BASE / 'closed.json').exists()
    prep=prepared(); spec=read(BASE / 'payload/payload.json'); shapes=read(BASE / 'payload/shapes.json')['shapes']
    generated,original=generate((ROOT / 'src/Lokad.Onnx/MathOps.cs').read_text(encoding='utf8-sig'))
    assert generated==(BASE / 'consumer/Block128.cs').read_text(encoding='utf8')
    assert original==(BASE / 'consumer/original-method.txt').read_text(encoding='utf8')
    assert len(shapes)==22 and spec['gates']==dict(process_max_min=1.10,geomean_candidate_baseline=.95,max_shape_candidate_baseline=1.05)
    collected=BASE / 'collected'; receipt=read(collected / 'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for name,wanted in receipt['files'].items(): assert pin(collected / name)==wanted,name
    assert receipt['payload']==prep['payload']==pin(collected / 'payload.json')
    state=read(collected / 'identity.json')
    assert state['complete'] and state['code']==0 and state['boot_time']==spec['boot_time']
    assert [r['name'] for r in state['runs']]==spec['jobs']
    remote_identities=[state['supervisor']]+[r['processes']['target'] for r in state['runs']]
    assert receipt['identities']==[read(collected / 'deployment.json')]+remote_identities[1:]
    confirmation=read(BASE / 'collection-transfer.json')
    assert confirmation['archive']==pin(BASE / 'results.tar.gz') and confirmation['receipt']==pin(collected / 'collection.json')
    live_check=json.loads(ssh(PRELUDE+f'''\nids={remote_identities!r}\nassert not any(live(i) for i in ids)\nprint(json.dumps(dict(terminal=True,identities=ids)))\n'''))
    assert live_check['terminal']
    resources=[]; local_identities=[]
    local=read(BASE / 'preparation.json')
    assert local['complete'] and local['code']==0 and [r['name'] for r in local['runs']]==['restore','build','validate-local']
    local_identities.append(local['supervisor'])
    for r in local['runs']:
        assert r['complete'] and r['code']==0 and r['seconds']<900
        assert r['preflight']['available']>=8*1024**3 and r['preflight']==r['preflight_observations'][-1]
        assert all(s['seconds']<900 and s['disk']>=20*1024**3 for s in r['preflight_observations'])
        samples=[json.loads(line) for line in (BASE / 'logs' / (r['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==r['samples']>0 and max(s['rss'] for s in samples)==r['peak_rss']
        for s in samples:
            assert s['seconds']<900 and s['rss']<4*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3 and s['output_bytes']<=1024**3
            assert s['rss']==sum(p['rss'] for p in s['members'])
            for p in s['members']: assert r['members'][str(p['pid'])]==p['birth'] and p['affinity']==[2]
        local_identities.extend(dict(pid=int(pid),birth=birth) for pid,birth in r['members'].items())
        resources.append(dict(location='Windows',name=r['name'],samples=len(samples),peak_rss=r['peak_rss'],seconds=r['seconds']))
    for i in local_identities: terminal(i)
    for r in state['runs']:
        assert r['complete'] and r['code']==0 and r['seconds']<900
        assert r['preflight']['available']>=8*1024**3 and r['preflight']['tmpfs']>=3*1024**3
        samples=[json.loads(line) for line in (collected / 'logs' / (r['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==r['samples']>0 and max(s['rss'] for s in samples)==r['peak_rss']
        for s in samples:
            assert s['seconds']<900 and s['rss']<2*1024**3 and s['available']>=1024**3 and s['tmpfs']>=1024**3 and s['artifacts']<=1024**3
            assert s['rss']==sum(p['rss'] for p in s['members']) and s['monitor_affinity']==[0]
            for p in s['members']:
                assert {k:p[k] for k in ['pid','birth']}==r['processes']['target'] and p['affinity']==[2]
                assert p['threads'] and all(t['affinity']==[2] for t in p['threads'])
        resources.append(dict(location='AMD',name=r['name'],samples=len(samples),peak_rss=r['peak_rss'],seconds=r['seconds']))
    expected=[(m,n,k,nonzero) for m in [0,2,4,6,8,10,16,32,64]
        for n in [0,1,9,31,63,64,127,128,129,255,256,257,511,512,513]
        for k in [0,1,7,8,15,31,32,33,63,64,65] for nonzero in [False,True]]
    expected += [(s['m'],s['n'],s['k'],nonzero) for s in shapes for nonzero in [False,True]]
    results={name:read(collected / 'output' / (name+'.json')) for name in spec['jobs']}
    local_result=read(BASE / 'output/validate-local.json')
    for name,r in [('validate-local',local_result),*results.items()]:
        assert r['passed'] and r['flags']==[] and r['processor_count']==1 and r['fma']
        assert r['core']==CORE and r['executable']==spec['consumer']['sha256'] and r['shapes']==pin(BASE / 'payload/shapes.json')['sha256']
        assert r['mode']==name.split('-')[0]
        owner=next(j for j in (local['runs'] if name=='validate-local' else state['runs']) if j['name']==name)
        assert r['pid']==(owner['worker']['pid'] if name=='validate-local' else owner['processes']['target']['pid'])
        assert r['runtime']==('10.0.12' if name=='validate-local' else '10.0.8')
        if name!='validate-local': assert r['avx512']
        if name.startswith('validate'):
            assert [(v['m'],v['n'],v['k'],v['nonzero']) for v in r['records']]==expected
            assert all(v['passed'] and v['values']==v['m']*v['k'] for v in r['records'])
            assert r['conditioning']==[]
        else:
            assert len(r['records'])==6*len(shapes)
            assert [(v['m'],v['n'],v['k']) for v in r['conditioning']]==[(s['m'],s['n'],s['k']) for s in shapes]
            assert all(v['calls']>=16 and v['seconds']>=1 for v in r['conditioning'])
    assert local_result['records']==results['validate']['records']
    digests={(r['m'],r['n'],r['k']):r['digest'] for r in results['validate']['records'] if not r['nonzero']}
    rows=[]
    for s in shapes:
        key=(s['m'],s['n'],s['k']); means={}
        for name in spec['jobs'][1:]:
            subset=[r for r in results[name]['records'] if (r['m'],r['n'],r['k'])==key]
            assert [r['block'] for r in subset]==list(range(6))
            for r in subset:
                assert r['iterations']==s['iterations'] and r['warm_calls']>=16 and r['warm_seconds']>=1 and r['digest']==digests[key]
                assert r['seconds']>0 and 0<=r['cpu_seconds']<=r['seconds']+.1
            means[name]=statistics.fmean(r['seconds']/r['iterations'] for r in subset)
        controls={role:max(means[role+'-a'],means[role+'-b'])/min(means[role+'-a'],means[role+'-b']) for role in ['baseline','candidate']}
        ratio=(means['candidate-a']+means['candidate-b'])/(means['baseline-a']+means['baseline-b'])
        rows.append(dict(**s,process_means=means,controls=controls,controls_passed=all(v<=1.10 for v in controls.values()),candidate_baseline=ratio))
    geo=math.exp(statistics.fmean(math.log(r['candidate_baseline']) for r in rows))
    worst=max(r['candidate_baseline'] for r in rows); controls=all(r['controls_passed'] for r in rows)
    analysis=dict(passed=True,eligible=controls and geo<=.95 and worst<=1.05,controls_passed=controls,
        geomean_candidate_baseline=geo,max_shape_candidate_baseline=worst,gates=spec['gates'],rows=rows,
        validation_cases=len(expected),validation_values=sum(r['values'] for r in results['validate']['records']),
        measured_blocks=4*22*6,resources=resources,local_identities=local_identities,remote_identities=remote_identities,
        scope='Synthetic complete-tile clearing, packing and multiplication on AMD; component eligibility only, no application or ORT timing claim.')
    save(BASE / 'analysis.json',analysis)
    files=dict(prep['files'])
    for folder in [collected,BASE / 'logs',BASE / 'output']:
        files.update({rel(p):pin(p) for p in folder.rglob('*') if p.is_file()})
    for p in BASE.iterdir():
        if p.is_file(): files[rel(p)]=pin(p)
    save(BASE / 'closed.json',dict(passed=True,files=files,analysis=pin(BASE / 'analysis.json'),local_identities=local_identities,remote_identities=remote_identities))
    print(json.dumps(dict(eligible=analysis['eligible'],controls=controls,geomean=geo,worst=worst,validation_cases=len(expected),
        resource_samples=sum(r['samples'] for r in resources),closed=pin(BASE / 'closed.json'))))


if __name__=='__main__': main()
