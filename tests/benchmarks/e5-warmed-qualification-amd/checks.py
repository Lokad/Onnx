import json
from protocol import read,pin


def normalized_body(value):
    """Normalize encoded widths while retaining every branch and exception target."""
    body=json.loads(value);ins=body['instructions'];assert ins and ins[-1]['opcode']=='ret'
    positions={i['offset']:n for n,i in enumerate(ins)};positions[ins[-1]['offset']+1]=len(ins)
    result=[]
    for index,item in enumerate(ins):
        op=item['opcode'];operand=item['operand']
        if op in ['ldc.i4','ldc.i4.s']:
            operand=int.from_bytes(bytes.fromhex(operand),'little',signed=True);op='ldc.i4'
        elif op!='break' and op.startswith(('br','beq','bne','bge','bgt','ble','blt','leave')):
            target=ins[index+1]['offset']+int.from_bytes(bytes.fromhex(operand),'little',signed=True)
            operand=positions[target];op=op.removesuffix('.s')
        elif op=='switch':
            raw=bytes.fromhex(operand);assert len(raw)%4==0
            operand=[positions[ins[index+1]['offset']+int.from_bytes(raw[n:n+4],'little',signed=True)] for n in range(0,len(raw),4)]
        result.append(dict(opcode=op,operand=operand))
    regions=[]
    for original in body['exceptions']:
        region=dict(original)
        for prefix in ['Try','Handler']:
            start=original[prefix+'Offset'];end=start+original[prefix+'Length']
            region[prefix+'Offset']=positions[start];region[prefix+'Length']=positions[end]-positions[start]
        if region['filter']!=-1:region['filter']=positions[region['filter']]
        regions.append(region)
    return dict(InitLocals=body['InitLocals'],MaxStackSize=body['MaxStackSize'],locals=body['locals'],exceptions=regions,instructions=result)


def consumer_inventory(value,spec,built):
    assert value['inventory_complete'] and len(value['observations'])==1
    row=value['observations'][0];assert row['assembly']=='ReleaseBenchmark.dll'
    assert row['public_surface_equal'] and row['compiler_rename'] is None and not row['added'] and not row['removed']
    assert row['before_sha256']==spec['previous_consumer']['sha256'] and row['after_sha256']==built['consumer']['sha256']
    assert row['method_flags_before']==row['method_flags_after']
    key,=row['differences'];assert key.startswith('Program::<Main>$::')
    old=normalized_body(row['normalized_methods'][key]);new=normalized_body(row['candidate_methods'][key])
    assert {k:v for k,v in old.items() if k!='instructions'}=={k:v for k,v in new.items() if k!='instructions'}
    changes=[]
    for index,(a,b) in enumerate(zip(old['instructions'],new['instructions'],strict=True)):
        if a==b:continue
        assert a['opcode']==b['opcode']=='ldc.i4'
        changes.append(dict(index=index,before=a['operand'],after=b['operand']))
    assert [(r['before'],r['after']) for r in changes]==[(780,1380),(600,1200)],changes
    assert row['unchanged_methods']==row['methods']-1
    return dict(passed=True,methods=row['methods'],unchanged_methods=row['unchanged_methods'],changes=changes,
        main_instructions=len(old['instructions']),branches_locals_exceptions_equal=True,implementation_flags_equal=True,
        previous_consumer=spec['previous_consumer'],consumer=built['consumer'],product_changed=False)

def check_result(base,name,spec,row):
    parts=name.split('-');mode=parts[0];role=parts[-1] if mode=='verify' else parts[-2]
    key='-'.join(parts[1:-1] if mode=='verify' else parts[1:-2])
    folder=base/name/'output';value=read(folder/'result.json')
    case=next(c for c in read(base/'cases.json')['cases'] if c['key']==key)
    assert value['passed'] and value['inputs_unchanged'] and value['held_outputs_unchanged'] and value['flags']=={}
    assert (value['role'],value['key'],value['mode'],value['pid'])==('ort' if role=='ort' else 'current',key,mode,row['child']['pid'])
    assert value['runtime']==('1.29.0' if role=='ort' else '10.0.8')
    if role!='ort':
        assert value['core']==spec['products'][role]['Lokad.Onnx.dll']['sha256']
        assert value['consumer']==read(base/'built.json')['consumer']['sha256']
    else:
        assert value['consumer']==pin(base/'tools/native.py')['sha256']
        assert value['native'] in [p['sha256'] for n,p in spec['external'].items() if 'onnxruntime_pybind11_state' in n]
    calls=3 if mode=='verify' else 1380
    assert value['calls']==len(value['clocks'])==calls
    for index,clock in enumerate(value['clocks']):
        assert clock['index']==index and clock['warmup']==(mode=='verify' or index<1200)
        assert type(clock['ticks']) is int and type(clock['frequency']) is int and clock['ticks']>0 and clock['frequency']>0
    assert len(value['arrays'])==len(case['outputs'])
    for actual,expected in zip(value['arrays'],case['outputs'],strict=True):
        assert (actual['name'],actual['shape'])==(expected['name'],expected['shape'])
        assert pin(folder/actual['file'])==dict(bytes=4*actual['values'],sha256=actual['sha256'])
        assert 0<=actual['max_scaled_error']<=1e-4
    return value
