"""Audit complete Where layouts, selected/native outputs, fixed exports and resources."""
import json
import collections
import math
import numpy as np
from protocol import JOBS,LIMITS,pin,read,save,check_sample
from prepare import ROOT,BASE,previous_closed

def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    folder=BASE/'collected';receipt=read(folder/'collection.json');state=read(folder/'identity.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json')
    payload=read(BASE/'payload.json')
    for name,wanted in read(BASE/'bundle/stage.json')['files'].items():
        assert pin(BASE/'bundle'/name)==wanted==payload['files'][name],name
    transfer=read(BASE/'collection-transfer.json');assert transfer['passed']
    assert transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(folder/'collection.json')
    assert prepared['archive']==pin(BASE/'payload.tar.gz') and prepared['stage']==pin(BASE/'bundle/stage.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0
    assert state['supervisor']==read(BASE/'deployment.json')
    assert state['ended']-state['started']<4*3600
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=0;peak=0;identities={(v['pid'],v['birth']) for v in receipt['identities']}
    assert (state['supervisor']['pid'],state['supervisor']['birth']) in identities
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        limit=LIMITS['build_preflight_available' if row['name'] in JOBS[:3] else 'preflight_available']
        assert row['preflight']['available']>=limit and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples'] and samples
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:
                assert (member['pid'],member['birth']) in identities
                assert row['members'][str(member['pid'])]==member['birth']
        assert max(s['rss'] for s in samples)==row['peak_rss']
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources+=len(samples);peak=max(peak,row['peak_rss'])
    built=read(folder/'built.json');assert built['passed']
    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
    for name,wanted in payload['product'].items(): assert pin(folder/'runtimes/current'/name)==wanted
    result=read(folder/'capture/result.json');manifest=read(BASE/'bundle/manifest.json')
    assert result['passed'] and result['no_performance_measurement'] and result['product_unchanged'] and result['graph_outputs_unchanged']
    assert result['pid']==state['runs'][-1]['child']['pid'] and result['runtime']=='10.0.8' and result['flags']=={}
    assert result['consumer_sha256']==built['consumer']['sha256']
    assert result['core_sha256']==payload['product']['Lokad.Onnx.dll']['sha256']
    assert result['manifest_sha256']==pin(BASE/'bundle/manifest.json')['sha256']
    census=read(BASE/'bundle/evidence/graph-observations.json')['graph']['wheres']
    assert len(census)==73 and [r['name'] for r in result['requests']]==[c['name'] for c in manifest['cases']]
    assert len(result['requests'])==2 and len(result['fixtures'])==manifest['fixture_count']==8
    seen={};errors=[];values=0;families=collections.Counter();eligible=0
    def metadata(info):
        assert info['dtype'] in ['Float','Int64','Int32','Bool'] and info['type'].startswith('Lokad.Onnx.')
        assert info['length']==math.prod(info['shape']) and len(info['strides'])==len(info['shape'])
        assert type(info['exact_dense']) is bool and type(info['reverse']) is bool
        assert all(type(x) is int and x>=0 for x in info['shape'])
        assert (info['scalar_bits'] is not None)==(info['length']==1)
    def array(value):
        metadata(value['tensor']);path=(folder/'capture'/value['file']).resolve()
        assert path.parent==(folder/'capture').resolve() and path.suffix=='.bin'
        wanted={k:value[k] for k in ['bytes','sha256']};assert pin(path)==wanted
        assert value['file'] not in seen;seen[value['file']]=wanted
        dtype={'Float':'<u4','Int64':'<i8','Int32':'<i4','Bool':'u1'}[value['tensor']['dtype']]
        data=np.fromfile(path,dtype=dtype).reshape(value['tensor']['shape'])
        assert data.size==value['tensor']['length'] and data.nbytes==value['bytes']
        if data.size==1:assert data.tobytes().hex()==value['tensor']['scalar_bits']
        if value['tensor']['dtype']=='Bool':assert np.isin(data,[0,1]).all()
        return data
    def fits(shape,output):
        return len(shape)<=len(output) and all(x==1 or x==y for x,y in zip(reversed(shape),reversed(output)))
    for request,case in zip(result['requests'],manifest['cases'],strict=True):
        assert request['selected_outputs_exact'] and request['inputs_unchanged'] and request['held_outputs_exact']
        observations=request['observations'];assert len(observations)==73
        assert [o['index'] for o in observations]==sorted({o['index'] for o in observations})
        assert request['graph_nodes']>max(o['index'] for o in observations)
        for observation,expected in zip(observations,census,strict=True):
            assert observation['name']==expected['name'] and observation['input_names']==expected['inputs']
            assert len(observation['inputs'])==3
            for info in [*observation['inputs'],observation['output']]:metadata(info)
            c,x,y=observation['inputs'];out=observation['output']
            assert c['dtype']=='Bool' and x['dtype']==y['dtype']==out['dtype']
            assert tuple(out['shape'])==np.broadcast_shapes(tuple(c['shape']),tuple(x['shape']),tuple(y['shape']))
            if expected['x_scalar']:
                assert x['dtype']=='Float' and x['length']==1
                assert x['scalar_bits']==np.array(expected['x_scalar']['value'],dtype='<f4').tobytes().hex()
            admitted=(x['dtype']=='Float' and all(t['exact_dense'] and not t['reverse'] for t in [c,x,y])
                and x['length']==1 and 1<=len(y['shape'])<=8 and out['shape']==y['shape']
                and fits(x['shape'],y['shape']) and fits(c['shape'],y['shape']))
            eligible+=int(admitted)
            family=json.dumps(dict(inputs=observation['inputs'],output=out,initial_scope_eligible=admitted),sort_keys=True)
            families[family]+=1
        assert len(request['outputs'])==2
        for output, (name,expected) in zip(request['outputs'],case['outputs'].items(),strict=True):
            actual=array(output)
            assert output['tensor']['shape']==expected['shape'] and output['tensor']['dtype']==expected['dtype']
            assert {k:output[k] for k in ['bytes','sha256']}=={k:expected[k] for k in ['bytes','sha256']}
            reference=case['native_outputs'][name];path=BASE/'bundle'/reference['file']
            assert pin(path)=={k:reference[k] for k in ['bytes','sha256']}
            native=np.load(path,allow_pickle=False);assert native.shape==actual.shape
            if expected['dtype']=='Float':
                managed=actual.view('<f4');assert np.isfinite(managed).all() and np.isfinite(native).all()
                error=float(np.max(np.abs(managed.astype(np.float64)-native.astype(np.float64))/np.maximum(1,np.abs(native.astype(np.float64)))))
                assert error<=1e-4
            else:assert np.array_equal(actual,native);error=0.0
            errors.append(dict(request=case['name'],output=name,max_error=error));values+=actual.size
    assert [f['name'] for f in result['fixtures']]==manifest['fixture_names']*2
    for fixture in result['fixtures']:
        request=next(r for r in result['requests'] if r['name']==fixture['request'])
        observed=next(r for r in request['observations'] if r['name']==fixture['name'])
        assert fixture['index']==observed['index']
        assert [v['tensor'] for v in fixture['inputs']]==observed['inputs'] and fixture['output']['tensor']==observed['output']
        c,x,y=[array(v) for v in fixture['inputs']];output=array(fixture['output'])
        # Float operands are uint32 here: compare selected bits, with no float arithmetic.
        assert np.array_equal(output,np.where(c.astype(bool),x,y))
    assert len(seen)==36 and result['exported_bytes']==sum(v['bytes'] for v in seen.values())<=manifest['export_cap_bytes']==32*1024**2
    assert {p.name for p in (folder/'capture').iterdir() if p.is_file()}==set(seen)|{'result.json'}
    analysis=dict(passed=True,no_performance_measurement=True,product_unchanged=True,graph_outputs_unchanged=True,
        product=payload['product'],consumer=built['consumer'],requests=2,where_observations=146,
        initial_scope_eligible=eligible,families=[dict(**json.loads(k),count=v) for k,v in families.items()],
        fixtures=8,export_files=len(seen),export_bytes=result['exported_bytes'],encoder_values=values,
        native_errors=errors,resources=resources,peak_rss=peak)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,product_unchanged=True,no_performance_measurement=True))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**{k:v for k,v in analysis.items() if k!='families'})))


if __name__=='__main__':main()
