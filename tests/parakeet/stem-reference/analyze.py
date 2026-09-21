"""Audit full reference arrays, independently reconstruct dots, retain all errors."""
from common import *
import math


def metrics(a,b,limit):
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    a=a.astype(np.float64);b=b.astype(np.float64);difference=a-b
    error=np.abs(difference)/np.maximum(1.,np.abs(b));index=int(error.argmax())
    return dict(values=int(a.size),max_scaled=float(error.flat[index]),failures=int(np.count_nonzero(error>limit)),
                rms=float(np.sqrt(np.mean(difference*difference))),max_absolute=float(np.abs(difference).max()),
                worst_index=index,coordinate=[int(i) for i in np.unravel_index(index,a.shape)],
                actual=float(a.flat[index]),reference=float(b.flat[index]))


def independent_dot(stage,coordinate,values,features,weights):
    if stage=='projection':
        batch,row,col=coordinate
        return math.fsum(a*b for a,b in zip(values['reshape'][batch,row].tolist(),weights['projection.weight'][:,col].tolist(),strict=True))
    i=int(stage.removeprefix('conv'));stride,pad,groups=GEOMETRY[i]
    previous={2:'relu0',3:'conv2',5:'relu3',6:'conv5'}
    x=features.astype(np.float64).transpose(0,2,1)[:,None] if i==0 else values[previous[i]]
    w=weights[f'conv{i}.weight'];bias=weights[f'conv{i}.bias'];batch,channel,row,col=coordinate
    first_channel=(channel//(w.shape[0]//groups))*w.shape[1]
    terms=[]
    for (ci,ky,kx),coefficient in np.ndenumerate(w[channel]):
        y=row*stride+ky-pad;z=col*stride+kx-pad
        terms.append(float(x[batch,first_channel+ci,y,z])*float(coefficient) if 0<=y<x.shape[2] and 0<=z<x.shape[3] else 0.)
    return math.fsum(terms)+float(bias[channel])


def main():
    spec=read(BASE/'manifest.json');verify(spec);state=read(BASE/'processes.json')
    assert state['complete'] and state['code']==0 and state['manifest']==pin(BASE/'manifest.json') and absent(state['supervisor'])
    assert [r['job'] for r in state['runs']]==JOBS
    samples=0;peak=0;identities=[state['supervisor']]
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and absent(row['worker']) and not row.get('error')
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['disk']>=LIMITS['disk']
        identities.append(row['worker'])
        raw=[json.loads(s) for s in (BASE/'process'/row['job']['id']/'samples.jsonl').read_text().splitlines()]
        assert len(raw)==row['samples'] and raw and row['peak_rss']==max(s['rss'] for s in raw)
        for s in raw:
            assert {k:s[k] for k in ('pid','birth')}==row['worker'] and s['affinity']==[2]
            assert s['seconds']<LIMITS['seconds'] and s['rss']<LIMITS['rss'] and s['available']>=LIMITS['available'] and s['disk']>=LIMITS['disk']
        samples+=len(raw);peak=max(peak,row['peak_rss'])
    values={};scalar_count=0;max_scalar=0.;records={}
    weights={key:np.load(ROOT/r['file'],allow_pickle=False).astype(np.float64) for key,r in spec['weights'].items()}
    for job in JOBS:
        folder=BASE/'outputs'/job['id'];result=read(folder/'result.json');records[job['id']]=result
        assert result['complete'] and result['job']==job and result['manifest']==state['manifest']
        assert result['inputs_unchanged'] and result['weights_unchanged'] and result['held_outputs_unchanged'] and not result['native_ort_loaded']
        run=next(r for r in state['runs'] if r['job']==job)
        assert {k:result['process'][k] for k in ('pid','birth')}==run['worker'] and result['process']['affinity']==[2]
        assert result['openblas_threads']==1 and result['numpy']==spec['numpy']
        for path,expected in result['libraries'].items():assert pin(path)==expected==spec['numerical_libraries'][path]
        assert (result['torch_config'] is not None)==(job['engine']=='torch')
        assert [r['name'] for r in result['outputs']]==list(STAGES)
        arrays={}
        for r in result['outputs']:
            assert r['shape']==SHAPES[r['name']] and r['dtype']=='float64'
            arrays[r['name']]=load_array(folder/r['file'],r)
        values[job['id']]=arrays
        features=np.load(ROOT/spec['inputs'][job['input']],allow_pickle=False)
        assert len(result['probes'])==1536
        for stage in [f'conv{i}' for i in CONVS]+['projection']:
            probes=[p for p in result['probes'] if p['stage']==stage]
            assert [p['index'] for p in probes]==coordinates(SHAPES[stage],stage)
            for probe in probes:
                coordinate=tuple(int(i) for i in np.unravel_index(probe['index'],SHAPES[stage]))
                assert probe['coordinate']==list(coordinate)
                expected=independent_dot(stage,coordinate,arrays,features,weights);actual=float(arrays[stage][coordinate])
                error=abs(actual-expected)/max(1.,abs(expected))
                assert actual==probe['actual'] and expected==probe['expected'] and error==probe['error'] and error<=REF_LIMIT
                scalar_count+=1;max_scalar=max(max_scalar,error)
        # Independently verify nonlinear/layout and bias stages exactly.
        for i in (0,3,6):assert np.array_equal(arrays[f'relu{i}'],np.maximum(arrays[f'conv{i}'],0.))
        assert arrays['reshape'].tobytes()==arrays['relu6'].transpose(0,2,1,3).reshape(1,74,4096).tobytes()
        assert arrays['stem'].tobytes()==(arrays['projection']+weights['projection.bias']).tobytes()
    assert scalar_count==9216
    agreement=[];repeats=[]
    for kind in ('native','managed','native-repeat'):
        for stage in STAGES:
            r=metrics(values['numpy-'+kind][stage],values['torch-'+kind][stage],REF_LIMIT)
            agreement.append(dict(input=kind,stage=stage,**r))
    for engine in ('numpy','torch'):
        for stage in STAGES:
            assert values[engine+'-native'][stage].tobytes()==values[engine+'-native-repeat'][stage].tobytes()
            repeats.append(dict(engine=engine,stage=stage,passed=True))
    qualified=all(r['failures']==0 for r in agreement)
    original=[]
    if qualified:
        for engine in ('managed','native'):
            for kind in ('native','managed'):
                r=spec['retained_stems'][engine+'-'+kind];actual=load_array(ROOT/r['file'],r)
                for reference in ('numpy','torch'):
                    expected=values[reference+'-'+kind]['stem'];m=metrics(actual,expected,1e-4)
                    # Scalar full-array max/count is independent of the vector metric.
                    errors=[abs(float(a)-float(b))/max(1.,abs(float(b))) for a,b in zip(actual.reshape(-1),expected.reshape(-1),strict=True)]
                    assert max(errors)==m['max_scaled'] and sum(e>1e-4 for e in errors)==m['failures']
                    original.append(dict(engine=engine,input=kind,reference_engine=reference,**m))
    result=dict(protocol=PROTOCOL,references_qualified=qualified,agreement=agreement,repeats=repeats,original=original,
                scalar_checks=scalar_count,max_scalar_error=max_scalar,resource_samples=samples,peak_rss=peak,
                calls=6,arrays=66,values=sum(a.size for stages in values.values() for a in stages.values()),
                bytes=sum(a.nbytes for stages in values.values() for a in stages.values()))
    write(BASE/'analysis.json',result)
    write(BASE/'closed.json',dict(protocol=PROTOCOL,references_qualified=qualified,identities=identities,
          files={rel(p):pin(p) for p in BASE.rglob('*') if p.is_file()},external_files=spec['files']))
    print(json.dumps(dict(qualified=qualified,max_reference_error=max(r['max_scaled'] for r in agreement),scalar_checks=scalar_count,
                         original=original,closed=pin(BASE/'closed.json'))))
    if not qualified:raise SystemExit(1)


if __name__=='__main__':main()
