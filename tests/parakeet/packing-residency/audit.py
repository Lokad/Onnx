from collections import Counter
import math
from run import ROOT,TOOLS,BASE,QUALIFICATION,pin,read,verify,save,terminal


def main():
    prepared=read(BASE/'prepared.json');assert prepared['passed'];verify(prepared['files'])
    state=read(BASE/'processes.json');assert state['complete'] and state['code']==0;terminal(state['supervisor'])
    assert [r['name'] for r in state['runs']]==[f'{role}-{mib}' for role in ('baseline','candidate') for mib in (256,512,2032)]
    graph=read(QUALIFICATION/'trace-output/graphs.json')['encoder'];weights={}
    for node in graph:
        if node['op']=='MatMul' and node['inputs'][1] in node['initializers']:
            name=node['inputs'][1];item=node['initializers'][name]
            assert item['dtype']=='Float' and len(item['shape'])==2
            weights[name]=item['shape']
    assert len(weights)==217 and sum(math.prod(s)*4 for s in weights.values())==2032*1024**2
    observations=[]
    for run in state['runs']:
        assert run['complete'] and run['passed'] and run['code']==0 and run['samples']>0;terminal(run['worker'])
        result=read(BASE/(run['name']+'.json'));assert result['passed'] and result['affinity']==4 and result['processor_count']==1 and result['runtime']=='.NET 10.0.12'
        role,mib=run['name'].split('-');mib=int(mib)
        for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','Census.dll')]:assert result[key]==pin(BASE/role/name)['sha256']
        names=set();total=0;shapes=Counter()
        for w in result['weights']:
            assert w['name'] not in names and w['name'] in weights;names.add(w['name'])
            assert w['shape']==weights[w['name']] and w['bytes']==math.prod(w['shape'])*4
            assert w['shape'][0] < (4096 if role=='baseline' else 4097)
            total+=w['bytes'];shapes[str(w['shape'])]+=1
        assert total==result['retained_bytes']==run['retained_bytes']<=result['budget']==mib*1024**2
        assert len(names)==run['weights']
        if mib==2032:
            expected={n for n,s in weights.items() if role=='candidate' or s[0]<4096}
            assert names==expected and total==(2032 if role=='candidate' else 1248)*1024**2
        if mib==256:assert len(names)==(28 if role=='candidate' else 37)
        samples=[__import__('json').loads(line) for line in (BASE/'logs'/(run['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==run['samples'] and max(s['rss'] for s in samples)==run['peak_rss']
        assert all(s['seconds']<180 and s['rss']<8*1024**3 and s['available']>=1024**3 and s['affinity']==[2] for s in samples)
        observations.append(dict(role=role,budget_mib=mib,retained_mib=total/1024**2,weights=len(names),shapes=dict(shapes),peak_rss=run['peak_rss'],samples=run['samples']))
    files=dict(prepared['files'])
    for p in BASE.rglob('*'):
        if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts):files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in TOOLS.iterdir():
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    assert not (BASE/'closed.json').exists()
    save(BASE/'closed.json',dict(passed=True,files=files,observations=observations,
         scope='Load-only residency and ownership; budgets above 256 MiB have no inference, application memory or speed qualification'))
    print(__import__('json').dumps(dict(passed=True,observations=observations,closed=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
