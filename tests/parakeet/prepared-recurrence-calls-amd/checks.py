"""Independently recompute every raw result, cache digest and native numerical bound."""
import math
import numpy as np
from protocol import pin, read


def array(folder,item,dtype=None):
    path=(folder/item['file']).resolve();assert path.is_relative_to(folder.resolve())
    assert pin(path)=={k:item[k] for k in ['bytes','sha256']}
    actual_dtype=dtype or {'Float':'<f4','Int32':'<i4','Int64':'<i8'}[item['dtype']]
    values=np.fromfile(path,dtype=actual_dtype)
    assert values.size==math.prod(item['shape'])==item['values'] and values.nbytes==item['bytes'] and np.isfinite(values).all()
    return values.reshape(item['shape'])


def qualify(base,name,payload,built,worker):
    role,mode=name.split('-');candidate=role=='candidate';folder=base/name/'output'
    result=read(folder/'result.json');spec=read(base/'spec.json');decoder=read(base/'decoder-spec.json')
    capture=read(base/'fixtures/result.json');native=read(base/'native/result.json')
    assert all(result[k] is True for k in ['passed','no_performance_measurement','original_graph_unchanged','held_outputs_unchanged','inputs_unchanged'])
    assert result['role']==role and result['mode']==mode and result['pid']==worker['child']['pid']
    assert result['affinity']==4 and result['processor_count']==1 and result['runtime']=='10.0.8' and result['avx512']==(mode=='512')
    assert result['flags']==({} if mode=='512' else {'DOTNET_EnableAVX512':'0'})
    for field,dll in [('core','Lokad.Onnx.dll'),('data','Lokad.Onnx.Data.dll')]:
        assert result[field]==spec['identities'][role][dll]['sha256']==payload['identities'][role][dll]['sha256']
    assert result['consumer']==built['consumer']['sha256'] and result['model_sha256']==decoder['model_sha256']
    for field,path in [('spec_sha256','spec.json'),('decoder_spec_sha256','decoder-spec.json'),('capture_sha256','fixtures/result.json'),('native_sha256','native/result.json')]:
        assert result[field]==pin(base/path)['sha256']
    assert result['original_node_sha256']==capture['original_node_sha256']
    expected_residencies=[]
    for graph_name in ['decoder','lstm-0','lstm-1']:
        matrices=spec['matrices'] if graph_name=='decoder' else []
        names=None if graph_name=='decoder' else decoder['nodes'][int(graph_name[-1])]['input_names'][1:3]
        recurrent=[w for w in spec['recurrent'] if names is None or w['name'] in names] if candidate else []
        weights=sorted(matrices+recurrent,key=lambda w:(w['kind'],w['name']))
        expected_residencies.append(dict(name=graph_name,budget=67108864,bytes=sum(w['bytes'] for w in weights),weights=weights))
    assert result['residencies']==expected_residencies
    assert result['routes']==[dict(name=n,prepared_route_proven=candidate,recovery_exact=True,held_outputs_unchanged=True) for n in ['decoder','lstm-0','lstm-1']]
    cache={};seen={};decoder_values=0;component_values=0;maximum=0.0;bitset=[]
    def actual(item):
        key=item['file'];wanted={k:item[k] for k in ['bytes','sha256']}
        if key not in cache:cache[key]=array(folder,item).reshape(-1);seen[key]=wanted
        assert seen[key]==wanted and cache[key].size==item['values']==math.prod(item['shape'])
        assert cache[key].dtype==np.dtype({'Float':'<f4','Int32':'<i4','Int64':'<i8'}[item['dtype']])
        bitset.append({k:item[k] for k in ['dtype','shape','bytes','sha256']})
        return cache[key].reshape(item['shape'])
    expected_steps=[(c['name'],repeat,step,s) for c in decoder['cases'] for repeat in [0,1] for step,s in enumerate(c['steps'])]
    assert len(result['decoder'])==len(expected_steps)==380
    for row,(case,repeat,step,expected) in zip(result['decoder'],expected_steps,strict=True):
        assert (row['name'],row['repeat'],row['step'],row['token'],row['duration'])==(case,repeat,step,expected['token'],expected['duration'])
        assert set(row['outputs'])==set(expected['outputs'])=={'outputs','prednet_lengths','output_states_1','output_states_2'}
        for key in sorted(row['outputs']):
            item=row['outputs'][key];reference=expected['outputs'][key]
            assert {k:item[k] for k in ['shape','dtype','bytes','sha256']}=={k:reference[k] for k in ['shape','dtype','bytes','sha256']}
            got=actual(item);decoder_values+=got.size
        logits=actual(row['outputs']['outputs']).reshape(-1)
        assert int(np.argmax(logits[:8193]))==row['token'] and int(np.argmax(logits[8193:]))==row['duration']
    # Component ordering preserves every actual captured call separately per node.
    expected_calls=[(index,repeat,c) for index in [0,1] for repeat in [0,1] for c in capture['calls'] if c['index']==index]
    references={(r['name'],r['step'],r['index'],r['output']):r['array'] for r in native['rows'] if r['repeat']==0}
    assert len(references)==1140 and len(result['calls'])==len(expected_calls)==760
    for row,(index,repeat,call) in zip(result['calls'],expected_calls,strict=True):
        assert (row['name'],row['step'],row['index'],row['repeat'])==(call['name'],call['step'],index,repeat)
        assert len(row['outputs'])==3
        for i,out in enumerate(row['outputs']):
            assert out['name']==call['output_names'][i];item=out['array'];expected=call['outputs'][i]
            assert item['dtype']=='Float' and all(item[k]==expected[k] for k in ['shape','bytes','sha256','values'])
            got=actual(item);selected=array(base/'fixtures',expected,'<f4')
            assert np.array_equal(got.view('<u4'),selected.view('<u4'))
            reference=array(base/'native',references[call['name'],call['step'],index,i],'<f4')
            assert got.shape==reference.shape
            errors=np.abs(got.astype(np.float64)-reference.astype(np.float64))/np.maximum(1.0,np.abs(reference.astype(np.float64)))
            error=float(errors.max(initial=0));worst=int(np.argmax(errors))
            assert error==out['max_error']<=1e-4 and worst==out['worst_index']
            maximum=max(maximum,error);component_values+=got.size
    assert component_values==1459200
    assert {p.name for p in folder.iterdir()}==set(seen)|{'result.json'}
    assert len(seen)==result['distinct_files'] and sum(v['bytes'] for v in seen.values())==result['tensor_bytes']
    assert result['tensor_bytes']<=128*1024**2
    # Stable normalized output receipt allows comparison across all four workers.
    import hashlib,json
    digest=hashlib.sha256(json.dumps(bitset,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(passed=True,role=role,mode=mode,result=pin(folder/'result.json'),output_digest=digest,
        decoder_executions=380,exact_decoder_arrays=1520,decoder_values=decoder_values,
        complete_calls=760,exact_component_arrays=2280,component_values=component_values,max_native_error=maximum,
        distinct_files=len(seen),tensor_bytes=result['tensor_bytes'],residencies=result['residencies'],
        prepared_routes_proven=3 if candidate else 0,no_performance_measurement=True)
