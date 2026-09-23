from protocol import read,pin

def check_result(base,name,spec,row):
    parts=name.split('-');mode=parts[0];role=parts[-1] if mode=='verify' else parts[-2]
    key='-'.join(parts[1:-1] if mode=='verify' else parts[1:-2])
    folder=base/name/'output';value=read(folder/'result.json')
    case=next(c for c in read(base/'cases.json')['cases'] if c['key']==key)
    assert value['passed'] and value['inputs_unchanged'] and value['held_outputs_unchanged'] and value['flags']=={}
    assert (value['role'],value['key'],value['mode'],value['pid'])==(role,key,mode,row['child']['pid'])
    assert value['runtime']==('10.0.8' if role=='current' else '1.29.0')
    if role=='current':
        assert value['core']==spec['product']['Lokad.Onnx.dll']['sha256']
        assert value['consumer']==read(base/'built.json')['consumer']['sha256']
    else:
        assert value['consumer']==pin(base/'tools/native.py')['sha256']
        assert value['native'] in [p['sha256'] for n,p in spec['external'].items() if 'onnxruntime_pybind11_state' in n]
    calls=3 if mode=='verify' else 120
    assert value['calls']==len(value['clocks'])==calls
    for index,clock in enumerate(value['clocks']):
        assert clock['index']==index and clock['warmup']==(mode=='verify' or index<60)
        assert type(clock['ticks']) is int and type(clock['frequency']) is int and clock['ticks']>0 and clock['frequency']>0
    assert len(value['arrays'])==len(case['outputs'])
    for actual,expected in zip(value['arrays'],case['outputs'],strict=True):
        assert (actual['name'],actual['shape'])==(expected['name'],expected['shape'])
        assert pin(folder/actual['file'])==dict(bytes=4*actual['values'],sha256=actual['sha256'])
        assert 0<=actual['max_scaled_error']<=1e-4
    return value
