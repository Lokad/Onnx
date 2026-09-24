"""Independent compiled-scope and observation checks; no model execution."""
from collections import Counter
import copy
import math

KINDS = [('attention-mask','self_attn/Where'), ('attention-cleanup','self_attn/Where_1'),
         ('convolution-mask','conv/Where'), ('attention-pad','self_attn/Pad'),
         ('convolution-pad','conv/depthwise_conv/Pad')]
TARGETS = {f'/layers.{layer}/{suffix}':kind for layer in range(24) for kind,suffix in KINDS}
BRANCHES = {'br','brfalse','brtrue','beq','bge','bgt','ble','blt','bne.un','bge.un','bgt.un','ble.un','blt.un','leave'}


def normalized_body(body, retained=None):
    """Express branches and exception regions as retained instruction ordinals."""
    rows = body['instructions']
    if retained is None: retained = list(range(len(rows)))
    assert retained == sorted(set(retained)) and retained[-1] == len(rows)-1
    index_at = {r['offset']:i for i,r in enumerate(rows)}
    # Main ends in ret. No branch into an instruction's operand is acceptable.
    assert rows[-1]['opcode'] == 'ret'
    index_at[rows[-1]['offset']+1] = len(rows)
    ordinals = {}; cursor = len(retained)
    for i in range(len(rows),-1,-1):
        if cursor and retained[cursor-1] == i: cursor -= 1
        ordinals[i] = cursor
    def position(offset): return ordinals[index_at[offset]]
    normalized = []
    for index in retained:
        row = rows[index]; op = row['opcode']; operand = row['operand']
        base = op.removesuffix('.s')
        if base in BRANCHES:
            width = 1 if op.endswith('.s') else 4
            raw = bytes.fromhex(operand); assert len(raw) == width
            target = row['offset']+1+width+int.from_bytes(raw,'little',signed=True)
            operand = position(target); op = base
        elif op == 'switch':
            raw = bytes.fromhex(operand); assert len(raw)%4 == 0
            end = row['offset']+5+len(raw)
            operand = [position(end+int.from_bytes(raw[i:i+4],'little',signed=True)) for i in range(0,len(raw),4)]
        normalized.append(dict(opcode=op,operand=operand))
    exceptions = []
    for item in body['exceptions']:
        value = {k:v for k,v in item.items() if k not in ['TryOffset','TryLength','HandlerOffset','HandlerLength','filter']}
        value.update(try_start=position(item['TryOffset']),try_end=position(item['TryOffset']+item['TryLength']),
            handler_start=position(item['HandlerOffset']),handler_end=position(item['HandlerOffset']+item['HandlerLength']),
            filter=-1 if item['filter'] == -1 else position(item['filter']))
        exceptions.append(value)
    return dict(instructions=normalized,exceptions=exceptions,
        **{k:v for k,v in body.items() if k not in ['instructions','exceptions']})


def consumer_scope(before, after, core_sha):
    after = copy.deepcopy(after); rows = after['instructions']; removed = set()
    initialize, = [i for i,r in enumerate(rows) if r['operand'] == 'MaskingConsumer::Void Initialize()']
    assert rows[initialize]['opcode'] == 'call'
    assert rows[initialize+1]['operand'] == 'System.Diagnostics.Stopwatch::Int64 GetTimestamp()'
    removed.add(initialize)
    save, = [i for i,r in enumerate(rows) if r['operand'] == 'MaskingConsumer::Void Save(System.String, Int32, System.String, Int32)']
    expected = [('ldloc.3',''), ('ldloc.s','0F'),
        ('callvirt','System.Collections.Generic.List`1[System.Object]::Int32 get_Count()'),
        ('ldloc.s','1A'), ('callvirt','Case::System.String get_Name()'), ('ldloc.s','17'),
        ('call','MaskingConsumer::Void Save(System.String, Int32, System.String, Int32)')]
    assert [(r['opcode'],r['operand']) for r in rows[save-6:save+1]] == expected
    removed.update(range(save-6,save+1))
    data, = [i for i,r in enumerate(rows) if r['opcode'] == 'ldstr' and r['operand'] == 'PARAKEET_MASKING_DATA_SHA']
    assert (rows[data+1]['opcode'],rows[data+1]['operand']) == ('call','System.Environment::System.String GetEnvironmentVariable(System.String)')
    rows[data]['operand'] = '065b7a7f28561a37174f9d38ac13ef77bc59c010702e2b64200e9542376eb4c5'
    removed.add(data+1)
    core, = [i for i,r in enumerate(rows) if r['opcode'] == 'ldstr' and r['operand'] == core_sha]
    rows[core]['operand'] = '672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35'
    assert normalized_body(before) == normalized_body(after,[i for i in range(len(rows)) if i not in removed]), \
        'Original consumer instructions, control flow, locals or exception regions changed'


def data_scope(before, after):
    assert before['InitLocals'] == after['InitLocals'] and before['MaxStackSize'] == after['MaxStackSize']
    assert before['locals'] == [] and before['exceptions'] == []
    assert after['locals'] == [dict(type='Lokad.Onnx.ParakeetMaskingProbe+Scope',IsPinned=False),
        dict(type='System.Collections.Generic.IReadOnlyDictionary`2[System.String,Lokad.Onnx.ITensor]',IsPinned=False)]
    rows = after['instructions']
    assert [r['opcode'] for r in rows[:3]] == ['ldarg.0','call','stloc.0']
    assert rows[1]['operand'] == 'Lokad.Onnx.ParakeetMaskingProbe::Scope Enter(Lokad.Onnx.GraphExecution)'
    body = [dict(r,offset=r['offset']-7) for r in rows if 7 <= r['offset'] < 120]
    assert body == before['instructions'][:-1], 'Original Data Execute body changed'
    tail = [r for r in rows if r['offset'] >= 120]
    assert [r['opcode'] for r in tail] == ['stloc.1','leave.s','ldloca.s','constrained.','callvirt','endfinally','ldloc.1','ret']
    assert tail[1]['operand'] == '0E' and tail[2]['operand'] == '00'
    assert tail[3]['operand'] == 'Lokad.Onnx.ParakeetMaskingProbe::Lokad.Onnx.ParakeetMaskingProbe+Scope'
    assert tail[4]['operand'] == 'System.IDisposable::Void Dispose()'
    assert after['exceptions'] == [dict(flags=2,TryOffset=7,TryLength=116,HandlerOffset=123,HandlerLength=14,filter=-1,caught=None)]


def layout(row):
    assert row is not None
    dims = row['Dimensions']; strides = row['Strides']
    assert len(dims) == len(strides) and all(type(v) is int and v >= 0 for v in dims)
    assert row['Length'] == math.prod(dims) and row['StorageLength'] >= 0
    assert row['RuntimeType'].startswith('Lokad.Onnx.')
    if row['ArrayBacked']:
        assert row['ArrayCount'] == row['StorageLength']
        assert 0 <= row['ArrayOffset'] <= row['ArrayLength']-row['ArrayCount']
    else: assert row['ArrayOffset'] == row['ArrayCount'] == row['ArrayLength'] == -1
    dense_strides = [math.prod(dims[i+1:]) for i in range(len(dims))]
    return dict(runtime_type=row['RuntimeType'],exact_dense=row['ExactDense'],reversed=row['Reversed'],
        row_major=strides == dense_strides,array_backed=row['ArrayBacked'],array_offset=row['ArrayOffset'],
        storage_matches_length=row['StorageLength'] == row['Length'])


def qualify(records, frames, graph_nodes):
    assert len(records) == 120 and len({r['Name'] for r in records}) == 120
    assert {r['Name'] for r in records} == set(TARGETS)
    assert [r['Index'] for r in records] == sorted({r['Index'] for r in records})
    output = []; t = frames
    for row in records:
        family = TARGETS[row['Name']]; assert row['Family'] == family
        node = graph_nodes[row['Index']]
        assert (row['Id'],row['Name'],row['Op'],row['InputNames'],row['OutputName']) == \
            (node['id'],node['name'],node['op'],node['inputs'],node['outputs'][0])
        assert len(row['Inputs']) == len(row['InputNames'])
        inputs = row['Inputs']; desc = [layout(v) if v is not None else None for v in inputs]
        result = layout(row['Output'])
        if row['Op'] == 'Where':
            assert len(inputs) == 3 and [v['Dtype'] for v in inputs] == ['Bool','Float','Float']
            condition = [1,1,t] if family == 'convolution-mask' else [1,1,t,t]
            shape = [1,1024,t] if family == 'convolution-mask' else [1,8,t,t]
            assert inputs[0]['Dimensions'] == condition and inputs[1]['Dimensions'] == []
            assert inputs[1]['ScalarBits'] == ('00401cc6' if family == 'attention-mask' else '00000000')
            assert inputs[2]['Dimensions'] == row['Output']['Dimensions'] == shape
            assert row['Output']['Dtype'] == 'Float'
            assert type(row['TrueCount']) is int and type(row['FalseCount']) is int
            assert row['TrueCount'] >= 0 and row['FalseCount'] >= 0
            assert row['TrueCount']+row['FalseCount'] == inputs[0]['Length']
            mask = 'all-false' if row['TrueCount'] == 0 else 'all-true' if row['FalseCount'] == 0 else 'mixed'
            assert row['Pads'] is None and row['Mode'] is None
        else:
            assert row['Op'] == 'Pad' and family in ['attention-pad','convolution-pad']
            shape = [1,8,t,2*t-1] if family == 'attention-pad' else [1,1024,t]
            expected = [1,8,t,2*t] if family == 'attention-pad' else [1,1024,t+8]
            pads = [0,0,0,1,0,0,0,0] if family == 'attention-pad' else [0,0,4,0,0,4]
            assert inputs[0]['Dimensions'] == shape and row['Output']['Dimensions'] == expected
            assert inputs[0]['Dtype'] == row['Output']['Dtype'] == 'Float'
            assert inputs[1]['Dtype'] in ['Int64','Int32'] and inputs[1]['Length'] == len(pads)
            assert row['Pads'] == pads and row['Mode'] == 'constant' and row['FillBits'] == '00000000'
            assert row['FillSource'] in ['input','default']
            if row['FillSource'] == 'input':
                assert inputs[2]['Dtype'] == 'Float' and inputs[2]['Length'] == 1 and inputs[2]['ScalarBits'] == row['FillBits']
            else: assert len(inputs) < 3 or inputs[2] is None
            assert row['TrueCount'] is None and row['FalseCount'] is None
            mask = None
        output.append(dict(name=row['Name'],family=family,frames=t,inputs=desc,output=result,mask=mask,
            true_count=row['TrueCount'],false_count=row['FalseCount']))
    assert Counter(r['family'] for r in output) == {kind:24 for kind,_ in KINDS}
    return output
