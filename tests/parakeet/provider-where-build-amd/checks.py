"""Exact declared Float-arm replacement; all other provider/tensor instructions survive."""
import copy
from il_body import normalized_body

HELPER = 'Lokad.Onnx.CPUExecutionProvider::Where::Lokad.Onnx.OpResult Where(Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ExecutionOptions)'
TENSOR = 'Lokad.Onnx.Tensor`1[T]::Where::Lokad.Onnx.Tensor`1[T] Where(Lokad.Onnx.Tensor`1[System.Boolean], Lokad.Onnx.Tensor`1[T], Lokad.Onnx.Tensor`1[T])'
TRY = 'Lokad.Onnx.UniformScalarWhere::Try::Boolean Try[T](Lokad.Onnx.Tensor`1[System.Boolean], Lokad.Onnx.Tensor`1[T], Lokad.Onnx.Tensor`1[T], Lokad.Onnx.Tensor`1[T] ByRef)'
ADDED = {TRY: 520}
BOOL = 'Lokad.Onnx.Tensor`1[System.Boolean]'
FLOAT = 'Lokad.Onnx.Tensor`1[System.Single]'


def canonical_local(item):
    op, value = item['opcode'], item['operand']
    for prefix in ['ldloca', 'ldloc', 'stloc']:
        if op == prefix or op == prefix + '.s':
            return dict(opcode=prefix, operand=int.from_bytes(bytes.fromhex(value), 'little'))
        if op.startswith(prefix + '.'):
            return dict(opcode=prefix, operand=int(op.split('.')[-1]))
    return dict(item)


def provider_body(original):
    before = normalized_body(original)
    assert len(before['instructions']) == 196 and not before['exceptions']
    assert before['MaxStackSize'] == 7 and before['locals'] == [
        dict(type='Lokad.Onnx.OpType', IsPinned=False), dict(type='Lokad.Onnx.TensorElementType', IsPinned=False)]
    result = copy.deepcopy(before)
    result['locals'].extend(dict(type=t, IsPinned=False) for t in [BOOL, FLOAT, FLOAT, FLOAT])
    old = before['instructions'][146:161]
    assert old[6] == dict(opcode='castclass', operand='::'+BOOL)
    assert old[8] == old[10] == dict(opcode='castclass', operand='::'+FLOAT)
    assert old[11]['opcode'] == 'call' and FLOAT+' Where(' in old[11]['operand']
    assert old[13]['opcode'] == 'call' and 'OpResult Success(' in old[13]['operand']
    # New locals: condition2, x3, y4, returned uniform5. All old locals survive.
    arm = [dict(opcode=op, operand=value) for op,value in [
        ('ldarg.0',''),('castclass','::'+BOOL),('stloc',2),
        ('ldarg.1',''),('castclass','::'+FLOAT),('stloc',3),
        ('ldarg.2',''),('castclass','::'+FLOAT),('stloc',4),
        ('ldloc',4),('callvirt',FLOAT+'::Int64 get_Length()'),('ldc.i4',4096),('conv.i8',''),('blt',182),
        ('ldloc',3),('callvirt',FLOAT+'::Int64 get_Length()'),('ldc.i4.1',''),('conv.i8',''),('bne.un',182),
        ('ldc.i4.5',''),('call','Lokad.Onnx.Profiler::Void StartOpStage(Lokad.Onnx.OpStage)'),
        ('ldloc',2),('ldloc',3),('ldloc',4),('ldloca',5),
        ('call',f'Lokad.Onnx.UniformScalarWhere::Boolean Try[Single]({BOOL}, {FLOAT}, {FLOAT}, {FLOAT} ByRef)'),
        ('brfalse',182),
        ('ldloc',0),('ldc.i4.1',''),('newarr','::Lokad.Onnx.ITensor'),('dup',''),('ldc.i4.0',''),
        ('ldloc',5),('stelem.ref',''),('call',old[13]['operand']),('ret',''),
        ('ldloc',0),('ldc.i4.1',''),('newarr','::Lokad.Onnx.ITensor'),('dup',''),('ldc.i4.0',''),
        ('ldloc',2),('ldloc',3),('ldloc',4),('call',old[11]['operand']),('stelem.ref',''),('call',old[13]['operand']),('ret','')]]
    assert len(arm) == 48 and arm[36] == dict(opcode='ldloc',operand=0)
    def move(target):
        assert not 146 < target < 161
        return target + 33 if target >= 161 else target
    instructions=[]
    for i, original in enumerate(before['instructions']):
        if 146 <= i < 161: continue
        item=canonical_local(original);op=item['opcode']
        if op=='switch':item['operand']=[move(t) for t in item['operand']]
        elif op!='break' and op.startswith(('br','beq','bne','bge','bgt','ble','blt','leave')):item['operand']=move(item['operand'])
        instructions.append(item)
    instructions[146:146]=arm
    result['instructions']=instructions
    return result


def inventory(value, measured, built, prior):
    assert value['inventory_complete'] and len(value['observations'])==2 and prior['passed']
    report=None
    for row,(name,count) in zip(value['observations'],[('Lokad.Onnx.dll',3189),('Lokad.Onnx.Data.dll',697)],strict=True):
        assert row['assembly']==name and row['methods']==len(row['normalized_methods'])==count
        assert row['before_sha256']==measured[name]['sha256'] and row['after_sha256']==built[name]['sha256']
        assert row['public_surface_equal'] and row['compiler_rename'] is None and not row['removed']
        before,after=row['method_flags_before'],row['method_flags_after']
        assert set(before)==set(row['normalized_methods'])
        assert all(after.get(key)==flag for key,flag in before.items())
        if name=='Lokad.Onnx.Data.dll':
            assert row['unchanged_methods']==697 and not row['differences'] and not row['added']
            assert not row['candidate_methods'] and before==after
            continue
        assert row['differences']==[HELPER] and row['unchanged_methods']==3188
        assert set(row['added'])==set(ADDED) and set(row['candidate_methods'])=={HELPER,*ADDED}
        assert after==before|ADDED and before[HELPER]==before[TENSOR]==0
        assert row['normalized_methods'][HELPER]==prior['helper_body'] and prior['helper_key']==HELPER
        actual=normalized_body(row['candidate_methods'][HELPER])
        actual['instructions']=[canonical_local(i) for i in actual['instructions']]
        assert actual==provider_body(prior['helper_body']), 'Provider IL differs from the declared Float-arm replacement'
        assert prior['uniform_key']==TRY
        helper=normalized_body(row['candidate_methods'][TRY])
        assert helper==normalized_body(prior['uniform_body']), 'Uniform helper must remain exactly qualified V3'
        report=dict(passed=True,original_core_methods=3189,candidate_core_methods=3190,unchanged_core_methods=3188,
            data_methods=697,changed_method=HELPER,original_provider_instructions=196,candidate_provider_instructions=229,
            float_arm_instructions_before=15,float_arm_instructions_after=48,existing_validation_and_nonfloat_edges_exact=True,
            generic_tensor_where_exact=True,all_existing_flags_exact=True,added_flags=ADDED,public_surface_equal=True,
            original_locals_preserved_with_four_declared_additions=True,qualified_uniform_helper_exact=True,
            new_helper_instructions=len(helper['instructions']))
    assert report is not None
    return report
