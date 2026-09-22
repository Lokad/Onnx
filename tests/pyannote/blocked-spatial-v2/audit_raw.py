"""Independently retain all numerical outcomes, including an unsuccessful prototype."""
import json
from prepare_raw import BASE,ROOT,MONITOR,pin,read,save,verify,rel,monitor


def main():
    assert not (BASE/'closed.json').exists()
    value=read(BASE/'verified.json');assert value['complete'];verify(value['files'])
    state=read(BASE/'preparation.json');assert state['complete']
    assert [r['name'] for r in state['runs']]==['restore','build','qualify-256']
    identities=[state['supervisor']];resources=[]
    for row in state['runs']:
        raw=row['name']=='qualify-256'
        assert row['complete'] and row['code'] in ([0,1] if raw else [0]) and row['seconds']<900
        assert row['preflight']['available']>=(12 if raw else 8)*1024**3
        samples=[json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for s in samples:
            assert s['seconds']<900 and s['rss']<8*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3 and s['output_bytes']<=1024**3
            assert s['rss']==sum(p['rss'] for p in s['members'])
            assert not raw or len(s['members'])<=1
            for p in s['members']:assert p['affinity']==[2] and row['members'][str(p['pid'])]==p['birth']
        identities.extend(dict(pid=int(p),birth=b) for p,b in row['members'].items())
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    for identity in identities:monitor.terminal(identity)
    result=read(BASE/'output/256.json');assert value['report']==pin(BASE/'output/256.json')
    assert result['cases']==len(result['observations'])==2648 and result['geometries']==312 and result['layout_cases']==331
    assert result['finite_kernel_cases']==sum(r['kernel'] for r in result['observations'])==2496
    assert result['nonfinite_fallback_cases']==sum(not r['kernel'] for r in result['observations'])==152
    assert all(r['kernel']==(not r['special']) for r in result['observations'])
    assert result['values']==sum(r['values'] for r in result['observations']) and result['rejected']==10 and result['owned_outputs']
    for field in ['scalar_differences','production_differences']:assert result[field]==sum(r[field] for r in result['observations'])
    assert len(result['supplemental'])==20 and sum(r['finite'] for r in result['supplemental'])==4
    assert all(r['kernel']==r['finite'] for r in result['supplemental'])
    assert result['failed_cases']==sum(r['scalar_differences']!=0 or r['production_differences']!=0 for r in result['observations'])+sum(r['differences']!=0 for r in result['supplemental'])
    assert result['passed']==value['passed']==(state['code']==0)==(result['failed_cases']==0)
    assert result['pid']==state['runs'][-1]['worker']['pid'] and not result['flags'] and result['lanes']==8
    analysis=dict(passed=result['passed'],qualification_complete=True,cases=result['cases'],values=result['values'],
        scalar_differences=result['scalar_differences'],production_differences=result['production_differences'],
        failed_cases=result['failed_cases'],resources=resources,identities=identities,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    files=dict(value['files']);files.update({rel(p):pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts)})
    save(BASE/'closed.json',dict(passed=result['passed'],qualification_complete=True,files=files,identities=identities,analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),passed=result['passed'],cases=result['cases'],failed_cases=result['failed_cases'],samples=sum(r['samples'] for r in resources))))


if __name__=='__main__':main()
