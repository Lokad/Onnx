"""Re-read every native comparison and refuse incomplete or changed contracts."""
from common import *
import numpy as np


def expected_rows(mode):
    rows=[]
    if mode=='e5':
        for case in CASES:
            fixture=read(E5/(case+'.json'))
            for policy in ('default','memory'):
                for context in ('facade','context'):
                    for step in range(3):rows.append((case,policy+'-'+context,step,'last_hidden_state',fixture['reference_file']))
    else:
        for model in read(REFERENCE/'manifest.json')['models']:
            for scenario in model['scenarios']:
                for step,item in enumerate(scenario['steps']):
                    for value in item['outputs']:rows.append((model['key'],scenario['name'],step,value['name'],value['file']))
    return rows


def inspect(spec,job):
    folder=BASE/'outputs'/job['id'];result=read(folder/'result.json')
    assert result['passed'] and result['mode']==job['mode'] and result['enabled'] is bool(job['enabled'])
    assert result['core_sha256']==spec['cores'][job['role']]['sha256'] and result['probe_sha256']==spec['runner']['sha256']
    assert result['runtime']=='10.0.12' and result['flags']==dict(LOKAD_ONNX_FINGERPRINT_STRINGS=str(job['enabled']))
    assert result['inputs_unchanged'] and result['held_outputs_unchanged']
    expected=expected_rows(job['mode'])
    assert [(r['model'],r['scenario'],r['step'],r['name'],r['reference_file']) for r in result['rows']]==expected
    assert len(expected)==(60 if job['mode']=='e5' else 106)
    assert len(result['graphs'])==(80 if job['mode']=='e5' else 11)
    for graph in result['graphs']:
        assert (graph['entries']>0 if job['enabled'] else graph['entries']==0) and isinstance(graph['fingerprint'],int)
    reference=ROOT/spec['references'][job['mode']];maximum=0.;count=0
    for index,row in enumerate(result['rows']):
        assert row['file']==str(index)+'.f32'
        path=folder/row['file'];assert pin(path)['sha256']==row['sha256']
        target=reference/row['reference_file'];assert pin(target)['sha256']==row['reference_sha256']
        wanted=np.fromfile(target,dtype='<f4') if job['mode']=='e5' else np.load(target,allow_pickle=False)
        actual=np.fromfile(path,dtype='<f4')
        assert wanted.dtype==actual.dtype==np.float32 and np.isfinite(wanted).all() and np.isfinite(actual).all()
        assert wanted.size==actual.size==row['values']==int(np.prod(row['shape']))
        shape=read(reference/(row['model']+'.json'))['shape'] if job['mode']=='e5' else list(wanted.shape)
        assert shape==row['shape']
        delta=np.abs(actual.astype(np.float64)-wanted.reshape(-1).astype(np.float64))/np.maximum(1.,np.abs(wanted.reshape(-1).astype(np.float64)))
        observed=float(delta.max(initial=0));failed=int(np.count_nonzero(delta>1e-4))
        assert failed==row['failed_values']==0 and abs(observed-row['max_scaled_error'])<=1e-15
        maximum=max(maximum,observed);count+=actual.size
    assert {p.name for p in folder.iterdir()}=={'result.json'}|{str(i)+'.f32' for i in range(len(expected))}
    return result,dict(job=job,arrays=len(expected),values=count,max_scaled_error=maximum,graph_checks=len(result['graphs']))


def main():
    spec=read(BASE/'manifest.json');verify(spec);state=read(BASE/'processes.json')
    assert state['complete'] and state['code']==0 and absent(state['supervisor'])
    assert [r['job'] for r in state['runs']]==JOBS
    identities=[state['supervisor']];samples=0;results={};summaries=[]
    for run in state['runs']:
        assert run['complete'] and run['code']==0 and absent(run['worker']) and run['role']==run['job']['id']
        identities.append(run['worker'])
        assert run['preflight']['available']>=LIMITS['preflight_available'] and run['preflight']['disk']>=LIMITS['disk']
        rows=[json.loads(line) for line in (BASE/'process'/run['role']/'samples.jsonl').read_text().splitlines()]
        assert len(rows)==run['samples']>0 and max(s['rss'] for s in rows)==run['peak_rss']
        assert 0<run['seconds']<LIMITS['seconds']
        for s in rows:
            assert s['seconds']<LIMITS['seconds'] and s['rss']<LIMITS['rss'] and s['available']>=LIMITS['available']
            assert s['disk']>=LIMITS['disk'] and s['affinity']==[2] and s['pid']==run['worker']['pid'] and s['birth']==run['worker']['birth']
        samples+=len(rows)
        result,summary=inspect(spec,run['job']);results[run['role']]=result;summaries.append(summary)
    comparisons=[]
    for mode in ('e5','shared'):
        canonical=results[mode+'-0-baseline']
        for enabled,role in ((0,'candidate'),(1,'baseline'),(1,'candidate')):
            key=f'{mode}-{enabled}-{role}';result=results[key]
            assert [r['sha256'] for r in result['rows']]==[r['sha256'] for r in canonical['rows']]
            assert [r['name'] for r in result['graphs']]==[r['name'] for r in canonical['graphs']]
            comparisons.append(dict(left=mode+'-0-baseline',right=key,arrays=len(result['rows']),bit_identical=True))
    assert sum(r['arrays'] for r in summaries)==664 and sum(r['values'] for r in summaries)==20003256
    result=dict(qualified=True,manifest=pin(BASE/'manifest.json'),workers=summaries,comparisons=comparisons,
        arrays=664,values=20003256,successful_executions=272,expected_input_failures=104,
        samples=samples,peak_rss=max(r['peak_rss'] for r in state['runs']),identities=identities)
    write(BASE/'analysis.json',result)
    files=dict(spec['files'])
    for path in BASE.rglob('*'):
        if path.is_file():files[rel(path)]=pin(path)
    write(BASE/'closed.json',dict(qualified=True,files=files,identities=identities,analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(qualified=True,arrays=664,values=20003256,workers=summaries,samples=samples,closure=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
