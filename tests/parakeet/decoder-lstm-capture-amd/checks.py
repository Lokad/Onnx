"""Recompute identities, all original controls, actual node geometry and native error."""
import hashlib
import math
import numpy as np
from protocol import pin, read


def array(folder,item):
    path=(folder/item['file']).resolve();assert path.is_relative_to(folder.resolve())
    assert pin(path)=={k:item[k] for k in ['bytes','sha256']}
    values=np.fromfile(path,dtype='<f4');assert values.size==math.prod(item['shape'])==item['values']
    assert values.nbytes==item['bytes'] and np.isfinite(values).all()
    return values.reshape(item['shape'])


def capture_review(base,payload,built,worker):
    folder=base/'capture/output';result=read(folder/'result.json');spec=read(base/'capture-spec.json')
    assert result['passed'] and result['no_performance_measurement'] and result['original_graph_unchanged'] and result['held_outputs_unchanged']
    assert result['pid']==worker['child']['pid'] and result['runtime']=='10.0.8' and result['avx512'] and result['vector_count']==8
    assert result['core']==spec['core']==payload['identities']['Lokad.Onnx.dll']['sha256']
    assert result['data']==spec['data']==payload['identities']['Lokad.Onnx.Data.dll']['sha256']
    assert result['executable']==built['consumer']['sha256'] and result['model_sha256']==spec['model_sha256']
    assert result['spec_sha256']==pin(base/'capture-spec.json')['sha256']
    assert len(result['original_node_sha256'])==64 and result['original_node_count']>2 and result['initializer_count']>=13
    outputs=['output_states_1','output_states_2','outputs','prednet_lengths'];assert result['original_outputs']==outputs
    expected_checks=[];expected_calls=[]
    for case in spec['cases']:
        for capture in [False,True]:
            for repeat in [0,1]:
                for step,item in enumerate(case['steps']):
                    assert sorted(item['outputs'])==outputs
                    expected_checks.append(dict(name=case['name'],step=step,capture=capture,repeat=repeat,
                        exact_outputs={k:v['sha256'] for k,v in item['outputs'].items()},inputs_unchanged=True,token=item['token'],duration=item['duration']))
        expected_calls.extend((case['name'],step,index) for step in range(len(case['steps'])) for index in [0,1])
    assert len(expected_checks)==760 and result['checks']==expected_checks
    calls=result['calls'];assert [(c['name'],c['step'],c['index']) for c in calls]==expected_calls and len(calls)==380
    cache={};seen={};values=0
    def actual(item):
        key=item['file'];wanted={k:item[k] for k in ['bytes','sha256']}
        if key not in cache:cache[key]=array(folder,item).reshape(-1);seen[key]=wanted
        assert wanted==seen[key] and cache[key].size==item['values']==math.prod(item['shape'])
        return cache[key].reshape(item['shape'])
    for key,item in result['tensors'].items():actual(item)
    for call in calls:
        node=spec['nodes'][call['index']]
        assert call['node']==node['name'] and call['opset']==node['opset'] and call['attributes']==node['attributes']
        assert call['input_names']==node['input_names'] and call['output_names']==node['output_names']
        assert len(call['inputs'])==7 and call['inputs'][4] is None and len(call['outputs'])==3
        for name,item,shape in zip(call['input_names'],call['inputs'],node['input_shapes'],strict=True):
            if not name:assert item is shape is None;continue
            assert item['shape']==shape;actual(item)
            key=('constant/'+name) if name in node['weights'] else (call['name']+'/'+str(call['step'])+'/'+name)
            assert result['tensors'][key]==item
            if name in node['weights']:assert {k:item[k] for k in ['bytes','shape','sha256']}==node['weights'][name]
        for name,item,shape in zip(call['output_names'],call['outputs'],node['output_shapes'],strict=True):
            assert item['shape']==shape;actual(item);values+=item['values']
            assert result['tensors'][call['name']+'/'+str(call['step'])+'/'+name]==item
    assert values==729600
    assert {p.name for p in folder.iterdir()}==set(seen)|{'result.json'}
    assert sum(v['bytes'] for v in seen.values())==result['tensor_bytes']<=64*1024**2
    assert len(seen)==result['distinct_files']
    return dict(passed=True,calls=380,original_decoder_executions=760,exact_original_arrays=3040,
        captured_output_arrays=1140,captured_output_values=values,unique_files=len(seen),tensor_bytes=result['tensor_bytes'],
        result=pin(folder/'result.json'),node_sha256=result['original_node_sha256'],no_performance_measurement=True)


def scaled(actual,expected):
    assert actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    errors=np.abs(actual.astype(np.float64)-expected.astype(np.float64))/np.maximum(1.0,np.abs(expected.astype(np.float64)))
    return float(errors.max(initial=0)),int(np.argmax(errors))


def native_review(base,payload,worker):
    result=read(base/'native/result.json');capture=read(base/'capture/output/result.json');spec=read(base/'capture-spec.json')
    assert result['passed'] and result['pid']==worker['child']['pid'] and result['affinity']==[2]
    assert result['onnxruntime']=='1.29.0' and result['numpy']=='2.2.4' and result['providers']==['CPUExecutionProvider']
    assert result['settings']==dict(intra=1,inter=1,sequential=True,all_optimizations=True,spinning=False)
    assert result['capture']==pin(base/'capture/output/result.json') and result['spec']==pin(base/'capture-spec.json')
    assert result['no_performance_measurement'] and result['held_outputs_unchanged'] and result['inputs_unchanged']
    assert result['flags']=={} and result['interpreter']==payload['interpreter']
    assert result['libraries'] and any('onnxruntime_pybind11_state' in name for name in result['libraries'])
    for name,wanted in result['libraries'].items():assert payload['external'][name]==wanted
    assert result['models']=={n['file']:n['model'] for n in spec['nodes']}
    expected_keys=[(c['name'],c['step'],c['index'],repeat,index) for c in capture['calls'] for repeat in [0,1] for index in range(3)]
    rows=result['rows'];assert [(r['name'],r['step'],r['index'],r['repeat'],r['output']) for r in rows]==expected_keys
    maximum=0.0;seen=set();repeat_hash={};total=0
    for offset,call in enumerate(capture['calls']):
        for row in rows[offset*6:(offset+1)*6]:
            got=array(base/'native',row['array']);wanted=array(base/'capture/output',call['outputs'][row['output']])
            # The native complete-call output is the reference denominator.
            error,index=scaled(wanted,got);assert error==row['max_error']<=1e-4 and index==row['worst_index']
            key=(row['name'],row['step'],row['index'],row['output'])
            assert key not in repeat_hash or repeat_hash[key]==row['array']['sha256'];repeat_hash[key]=row['array']['sha256']
            maximum=max(maximum,error);total+=got.size;seen.add(row['array']['file'])
    assert len(rows)==2280 and total==1459200 and result['max_error']==maximum
    assert {p.name for p in (base/'native').iterdir()}==seen|({'result.json','review.json'} if (base/'native/review.json').exists() else {'result.json'})
    return dict(passed=True,calls=760,arrays=2280,values=total,max_error=maximum,native_repeat_exact=True,
                result=pin(base/'native/result.json'),no_performance_measurement=True)
