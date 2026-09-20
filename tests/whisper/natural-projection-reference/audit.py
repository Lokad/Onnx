"""Recompute every comparison and independent scalar reference checks after exit."""
import argparse
from common import *

def resource_audit(state,samples,spec):
    assert state['complete'] is True and state['code']==0 and not state.get('error')
    assert state['seconds']<spec['limits']['seconds'] and state['started']<=state['ended']
    assert state['preflight']['available']>=spec['limits']['preflight_available'] and state['preflight']['disk']>=spec['limits']['disk']
    assert state['worker']['birth']>=state['supervisor']['birth']
    assert samples and all(a['seconds']<=b['seconds'] for a,b in zip(samples,samples[1:]))
    for row in samples:
        assert row['pid']==state['worker']['pid'] and row['birth']==state['worker']['birth']
        assert 0<=row['seconds']<=state['seconds'] and row['rss']<spec['limits']['rss'] and row['available']>=spec['limits']['available'] and row['affinity']==[2]
    assert max([samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[state['seconds']-samples[-1]['seconds']])<10
    return dict(samples=len(samples),peak_sampled_rss=max(r['rss'] for r in samples),minimum_available=min(r['available'] for r in samples),seconds=state['seconds'])

def audit(base):
    spec=read(base/'manifest.json');verify(spec['files']);assert spec['limits']==LIMITS and spec['projections']==PROJECTIONS and len(spec['jobs'])==64
    state=read(base/'campaign.json');samples=[json.loads(s) for s in (base/'samples.jsonl').read_text().splitlines()]
    resource=resource_audit(state,samples,spec);assert absent(state['supervisor']) and absent(state['worker'])
    assert state['manifest']==pin(base/'manifest.json');result=read(base/'result.json');assert result['complete'] and result['manifest']==state['manifest']
    for runtime in [result['runtime'],result['runtime_after']]:
        assert runtime['numpy']==spec['numpy'] and runtime['blas_threads']==1 and runtime['affinity']==[2]
        assert runtime['pid']==state['worker']['pid'] and runtime['birth']==state['worker']['birth']
        for path,identity in runtime['loaded'].items():assert pin(path)==identity
    assert [r['id'] for r in result['rows']]==[j['id'] for j in spec['jobs']]
    rows=[];groups=[];first={};bytes_saved=0;all_checks=0;max_fsum_error=0.;expected=set()
    for request in range(8):
        for projection in PROJECTIONS:
            actuals={};refs={};metadata=[]
            jobs=[(j,r) for j,r in zip(spec['jobs'],result['rows']) if j['request']==request and j['projection']==projection]
            assert [j['cell'] for j,r in jobs]==CELLS
            w,b=weights(projection)
            for job,row in jobs:
                path=base/row['file'];assert row==read(base/'references'/(job['id']+'.json'))
                assert path==base/'references'/(job['id']+'.f64') and pin(path)==row['pin']
                shape=job['actual']['shape'][1:];assert row['shape']==shape and row['pin']['bytes']==math.prod(shape)*8
                ref=np.fromfile(path,dtype='<f8').reshape(shape);assert np.isfinite(ref).all();bytes_saved+=ref.nbytes
                actual=load(job['actual']);x=load(job['input']);error=metrics(actual-ref,np.maximum(1,np.abs(ref)));assert error==row['local_error']
                coords=coordinates(ref.shape,[error['scaled_index'],error['absolute_index']]);checks=coordinate_checks(x,w,b,ref,coords)
                assert checks==row['checks'];all_checks+=len(checks);max_fsum_error=max(max_fsum_error,max(c['error'] for c in checks))
                # Independently conservative envelope; no second BLAS reference run.
                magnitude=x.shape[1]*float(np.abs(x).max())*float(np.abs(w).max())+(0 if b is None else float(np.abs(b).max()))
                uniform=math.nextafter(2*gamma(x.shape[1])*magnitude,math.inf)
                assert 0<=row['reference_scaled_bound_max']<=row['reference_error_bound_max']<=uniform
                assert max(c['error'] for c in checks)<=row['reference_error_bound_max']+4*max(math.ulp(c['fsum']) for c in checks)
                key=(job['features'],projection,job['cell'])
                if request in [0,1]:first[key]=row['pin']
                if request in [6,7]:assert first[key]==row['pin'],'Repeat reference changed'
                actuals[job['cell']]=actual;refs[job['cell']]=ref;metadata.append(job)
                rows.append(dict(request=request,name=job['name'],features=job['features'],cell=job['cell'],projection=projection,
                    **{k:row[k] for k in ['local_error','reference_error_bound_max','reference_scaled_bound_max']},independent_checks=len(checks)))
                expected|={job['id']+'.json',job['id']+'.f64'}
            groups.append(dict(request=request,name=metadata[0]['name'],features=metadata[0]['features'],projection=projection,
                common_denominator='max(1,abs(actual NN)); all input and reference arrays retain all 1500 frames',contrasts=decompose(actuals,refs)))
    assert bytes_saved==spec['reference_bytes']==2457600000 and {p.name for p in (base/'references').iterdir()}==expected
    return dict(passed=True,manifest=pin(base/'manifest.json'),prior_receipt=spec['prior_receipt'],resources=resource,reference_arrays=64,reference_bytes=bytes_saved,
        scalar_checks=all_checks,max_fsum_error=max_fsum_error,repeats_exact=True,rows=rows,groups=groups)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    value=audit(Path(a.artifact).resolve());write(Path(a.output),value);print(json.dumps({k:v for k,v in value.items() if k not in ['rows','groups']}))
