"""Independently close complete captured operands, native bounds and resource evidence."""
import importlib.util
import json
import math
from run import ROOT, BASE, CORE, DATA, JOBS, pin, read, save, verify, monitor


def main():
    assert not (BASE/'closed.json').exists()
    value=read(BASE/'verified.json'); assert value['passed']
    for name in ['inputs.json','binaries.json']: verify(read(BASE/name)['files'])
    spec=importlib.util.spec_from_file_location('lstm_fixture_resources',ROOT/'tests/parakeet/portable-models/common.py')
    common=importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    resources=common.resources(BASE,'controller.json',JOBS)
    state=read(BASE/'controller.json'); capture=read(BASE/'output/result.json'); native=read(BASE/'native/result.json')
    assert value['capture']==pin(BASE/'output/result.json') and value['native']==pin(BASE/'native/result.json')
    assert capture['passed'] and native['passed'] and native['capture']==value['capture']
    assert capture['core']==CORE and capture['data']==DATA and capture['no_performance_measurement']
    assert capture['held_outputs_unchanged'] and capture['original_graph_unchanged'] and capture['captured_output_bindings_extended']
    assert capture['pid']==state['runs'][2]['worker']['pid'] and native['pid']==state['runs'][3]['worker']['pid']
    assert capture['executable']==pin(BASE/'consumer/bin/Release/net10.0/LstmCapture.dll')['sha256']
    assert capture['runtime']=='10.0.12' and capture['vector_count']==8 and not capture['avx512']
    cases=read(BASE/'capture-spec.json')['cases']; calls=capture['calls']
    assert [(c['name'],c['index']) for c in calls]==[(c['name'],i) for c in cases for i in range(4)]
    paths={}
    for item in capture['tensors'].values():
        assert item['values']==math.prod(item['shape']) and item['bytes']==item['values']*4
        expected={k:item[k] for k in ['bytes','sha256']}; assert pin(BASE/'output'/item['file'])==expected
        paths[item['file']]=expected
    assert len(paths)==capture['distinct_files'] and sum(p['bytes'] for p in paths.values())==capture['tensor_bytes']<=128*1024**2
    for call in calls:
        size=60 if call['index']==0 else 256
        assert call['attributes']==dict(direction='bidirectional',hidden_size=128) and call['opset']==17
        assert len(call['inputs'])==7 and len(call['outputs'])==3 and call['inputs'][4] is None
        expected=[[589,1,size],[2,512,size],[2,512,128],[2,1024],None,[2,1,128],[2,1,128]]
        for item,shape in zip(call['inputs'],expected,strict=True): assert (None if item is None else item['shape'])==shape
        assert [r['shape'] for r in call['outputs']]==[[589,2,1,128],[2,1,128],[2,1,128]]
        for names,items in [(call['input_names'],call['inputs']),(call['output_names'],call['outputs'])]:
            for name,item in zip(names,items,strict=True):
                assert (item is None)==(name=='')
                if item is not None: assert item==capture['tensors'][call['name']+'/'+name]
    assert len(capture['checks'])==12
    assert [(c['name'],c['capture'],c['repeat']) for c in capture['checks']]==[(c['name'],flag,repeat) for c in cases for flag in [False,True] for repeat in [0,1]]
    for c in capture['checks']:
        assert c['exact'] and c['input_unchanged']
        row,=[r for r in cases if r['name']==c['name']]; assert c['sha256']==row['selected_sha256']
    import numpy as np
    assert len(native['reports'])==36 and len(native['graphs'])==3 and native['version']=='1.29.0'
    values=0; maximum=0.
    for i,row in enumerate(native['reports']):
        call=calls[i//3]; slot=i%3
        assert (row['case'],row['index'],row['node'],row['slot'])==(call['name'],call['index'],call['node'],slot)
        assert row['exact_repeat'] and row['input_unchanged']
        ref=row['reference']; assert pin(BASE/'native'/ref['file'])=={k:ref[k] for k in ['bytes','sha256']}
        assert ref['shape']==call['outputs'][slot]['shape']
        a=np.fromfile(BASE/'output'/call['outputs'][slot]['file'],dtype='<f4').astype('float64')
        b=np.fromfile(BASE/'native'/ref['file'],dtype='<f4').astype('float64')
        assert np.isfinite(a).all() and np.isfinite(b).all() and a.shape==b.shape
        error=np.abs(a-b)/np.maximum(1.,np.abs(b)); comparison=dict(values=int(a.size),failed=int(np.count_nonzero(error>1e-4)),maximum=float(error.max(initial=0)))
        assert comparison==row['comparison'] and comparison['failed']==0
        values+=len(a); maximum=max(maximum,comparison['maximum'])
    assert maximum==native['maximum']
    for row in native['graphs']:
        assert row['exact_repeat'] and row['input_unchanged']
        for name in ['selected','retained']: assert row[name]['failed']==0 and row[name]['maximum']<=1e-4
    analysis=dict(passed=True,calls=12,outputs=36,values=values,native_maximum=maximum,tensor_bytes=capture['tensor_bytes'],
        local_vector_count=8,actual_amd_pending=True,resources=resources['resources'],no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json',dict(passed=True,files=files,identities=resources['identities'],local_inputs=read(BASE/'inputs.json')['files']))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__': main()
