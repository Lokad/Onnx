"""Independent compiled-scope and observation checks; no model execution."""
from collections import Counter
import copy
import math

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
    initialize, = [i for i,r in enumerate(rows) if r['operand'] == 'ProjectionConsumer::Void Initialize()']
    assert rows[initialize]['opcode'] == 'call'
    assert rows[initialize+1]['operand'] == 'System.Diagnostics.Stopwatch::Int64 GetTimestamp()'
    removed.add(initialize)
    save, = [i for i,r in enumerate(rows) if r['operand'] == 'ProjectionConsumer::Void Save(System.String, Int32, System.String, Int32)']
    expected = [('ldloc.3',''), ('ldloc.s','0F'),
        ('callvirt','System.Collections.Generic.List`1[System.Object]::Int32 get_Count()'),
        ('ldloc.s','1A'), ('callvirt','Case::System.String get_Name()'), ('ldloc.s','17'),
        ('call','ProjectionConsumer::Void Save(System.String, Int32, System.String, Int32)')]
    assert [(r['opcode'],r['operand']) for r in rows[save-6:save+1]] == expected
    removed.update(range(save-6,save+1))
    data, = [i for i,r in enumerate(rows) if r['opcode'] == 'ldstr' and r['operand'] == 'PARAKEET_PROJECTION_DATA_SHA']
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
    assert after['locals'] == [dict(type='Lokad.Onnx.ParakeetProjectionProbe+Scope',IsPinned=False),
        dict(type='System.Collections.Generic.IReadOnlyDictionary`2[System.String,Lokad.Onnx.ITensor]',IsPinned=False)]
    rows = after['instructions']
    assert [r['opcode'] for r in rows[:3]] == ['ldarg.0','call','stloc.0']
    assert rows[1]['operand'] == 'Lokad.Onnx.ParakeetProjectionProbe::Scope Enter(Lokad.Onnx.GraphExecution)'
    body = [dict(r,offset=r['offset']-7) for r in rows if 7 <= r['offset'] < 120]
    assert body == before['instructions'][:-1], 'Original Data Execute body changed'
    tail = [r for r in rows if r['offset'] >= 120]
    assert [r['opcode'] for r in tail] == ['stloc.1','leave.s','ldloca.s','constrained.','callvirt','endfinally','ldloc.1','ret']
    assert tail[1]['operand'] == '0E' and tail[2]['operand'] == '00'
    assert tail[3]['operand'] == 'Lokad.Onnx.ParakeetProjectionProbe::Lokad.Onnx.ParakeetProjectionProbe+Scope'
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


def qualify(call, graph, frequency):
    assert frequency == 1000000000 and call['GraphNodes'] == len(graph) == 2856
    walls = call['Nodes']; assert len(walls) == len(graph)
    previous = 0
    wall_by_id = {}
    for row,node in zip(walls,graph):
        assert row['NodeId'] == node['id'] and row['NodeId'] not in wall_by_id
        assert previous <= row['StartTicks'] < row['EndTicks']
        previous = row['EndTicks']; wall_by_id[row['NodeId']] = row
    targets = {n['name']:n for n in graph if n['op'] == 'MatMul' and len(n['constant_inputs']) == 2
        and n['constant_inputs'][1] and len(n['constant_inputs'][1]['dims']) == 2
        and min(n['constant_inputs'][1]['dims']) >= 1024 and n['constant_inputs'][1]['type'] == 'Float'}
    assert len(targets) == len(call['Records']) == 217
    assert {r['Name'] for r in call['Records']} == targets.keys()
    assert [r['Index'] for r in call['Records']] == sorted({r['Index'] for r in call['Records']})
    result = []
    for row in call['Records']:
        node = targets[row['Name']]
        assert graph[row['Index']] == node
        assert (row['Id'],row['InputNames'],row['OutputName']) == (node['id'],node['inputs'],node['outputs'][0])
        a,b,output = row['A'],row['B'],row['Output']
        descriptions = [layout(r) for r in [a,b,output]]
        assert a['Dimensions'][-1] == b['Dimensions'][0] and len(b['Dimensions']) == 2
        assert b['Dimensions'] == node['constant_inputs'][1]['dims'] == [row['K'],row['N']]
        assert a['Dimensions'][-2] == row['M']
        assert output['Dimensions'] == a['Dimensions'][:-1]+[row['N']]
        assert row['RowGuardAllows'] == (row['M']%2 == 0 or row['M']%3 == 0)
        assert row['UseSimd'] and row['UseIntrinsics'] and row['Fma'] and row['Avx512']
        assert row['DegreeOfParallelism'] == 1
        assert 0 <= row['ScratchBytes'] <= call['ScratchBytes'] and 0 <= row['CopyBytes'] <= call['CopyBytes']
        wall = wall_by_id[row['Id']]
        assert row['StartTicks'] <= wall['StartTicks'] < wall['EndTicks'] <= row['EndTicks']
        mapped = row['Prepared'] is not None
        if mapped:
            layout(row['Prepared'])
            assert row['MappedSource'] == node['inputs'][1] and row['SourceReferenceMatches']
            assert row['Prepared']['Dimensions'] == b['Dimensions']
        else:
            assert row['MappedSource'] is None and not row['SourceReferenceMatches']
        # Report the measured scratch/copy counters independently of inferred routing.
        result.append(dict(name=row['Name'],m=row['M'],k=row['K'],n=row['N'],mapped=mapped,
            row_guard_allows=row['RowGuardAllows'],scratch_bytes=row['ScratchBytes'],copy_bytes=row['CopyBytes'],
            node_ticks=wall['EndTicks']-wall['StartTicks'],interval_ticks=row['EndTicks']-row['StartTicks'],
            layouts=descriptions))
    assert sum(r['scratch_bytes'] for r in result) <= call['ScratchBytes']
    assert sum(r['copy_bytes'] for r in result) <= call['CopyBytes']
    return result

