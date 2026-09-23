"""Permit only the declared float guard, one new local and one private helper."""
import copy
from il_body import normalized_body

OWNER = 'Lokad.Onnx.Tensor`1[T]'
HELPER = OWNER + '::Where::Lokad.Onnx.Tensor`1[T] Where(Lokad.Onnx.Tensor`1[System.Boolean], Lokad.Onnx.Tensor`1[T], Lokad.Onnx.Tensor`1[T])'
TRY = OWNER + '::TryUniformScalarWhere::Boolean TryUniformScalarWhere(Lokad.Onnx.Tensor`1[System.Boolean], Lokad.Onnx.Tensor`1[T], Lokad.Onnx.Tensor`1[T], Lokad.Onnx.Tensor`1[T] ByRef)'
ADDED = {TRY: 520}


def operand(key):
    owner, _, signature = key.split('::', 2)
    return owner + '::' + signature


def canonical_local(item):
    op, value = item['opcode'], item['operand']
    for prefix in ['ldloca', 'ldloc', 'stloc']:
        if op == prefix or op == prefix + '.s':
            return dict(opcode=prefix, operand=int.from_bytes(bytes.fromhex(value), 'little'))
        if op.startswith(prefix + '.'):
            return dict(opcode=prefix, operand=int(op.split('.')[-1]))
    return dict(item)


def guarded_body(original):
    before = normalized_body(original)
    assert len(before['instructions']) == 79 and not before['exceptions']
    assert before['MaxStackSize'] == 4 and len(before['locals']) == 9
    result = copy.deepcopy(before)
    result['locals'].insert(0, dict(type=OWNER, IsPinned=False))
    insertion = [dict(opcode=op, operand=value) for op, value in [
        ('ldtoken', OWNER + '::T'),
        ('call', 'System.Type::System.Type GetTypeFromHandle(System.RuntimeTypeHandle)'),
        ('ldtoken', '::System.Single'),
        ('call', 'System.Type::System.Type GetTypeFromHandle(System.RuntimeTypeHandle)'),
        ('call', 'System.Type::Boolean op_Equality(System.Type, System.Type)'),
        ('brfalse', 16), ('ldarg.0', ''), ('ldarg.1', ''), ('ldarg.2', ''),
        ('ldloca', 0), ('call', operand(TRY)), ('brfalse', 16), ('ldloc', 0), ('ret', '')]]
    instructions = []
    for original in before['instructions']:
        item = canonical_local(original); op = item['opcode']
        if op in ['ldloca', 'ldloc', 'stloc']: item['operand'] += 1
        elif op != 'break' and op.startswith(('br', 'beq', 'bne', 'bge', 'bgt', 'ble', 'blt', 'leave')):
            assert item['operand'] >= 2
            item['operand'] += len(insertion)
        else: assert op != 'switch'
        instructions.append(item)
    instructions[2:2] = insertion
    result['instructions'] = instructions
    return result


def inventory(value, measured, built, prior):
    assert value['inventory_complete'] and len(value['observations']) == 2 and prior['passed']
    report = None
    for row, (name, count) in zip(value['observations'], [('Lokad.Onnx.dll', 3189), ('Lokad.Onnx.Data.dll', 697)], strict=True):
        assert row['assembly'] == name and row['methods'] == len(row['normalized_methods']) == count
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert row['public_surface_equal'] and row['compiler_rename'] is None and not row['removed']
        before, after = row['method_flags_before'], row['method_flags_after']
        assert set(before) == set(row['normalized_methods'])
        assert all(after.get(key) == flag for key, flag in before.items())
        if name == 'Lokad.Onnx.Data.dll':
            assert row['unchanged_methods'] == 697 and not row['differences'] and not row['added']
            assert not row['candidate_methods'] and before == after
            continue
        assert row['differences'] == [HELPER] and row['unchanged_methods'] == 3188
        assert set(row['added']) == set(ADDED) and set(row['candidate_methods']) == {HELPER, *ADDED}
        assert after == before | ADDED and before[HELPER] == 0
        assert row['normalized_methods'][HELPER] == prior['helper_body'] and prior['helper_key'] == HELPER
        actual = normalized_body(row['candidate_methods'][HELPER])
        actual['instructions'] = [canonical_local(i) for i in actual['instructions']]
        assert actual == guarded_body(prior['helper_body'])
        helper = normalized_body(row['candidate_methods'][TRY])
        assert not helper['exceptions']
        calls = [i['operand'] for i in helper['instructions'] if i['opcode'] in ['call', 'callvirt']]
        assert any('IndexOf[Boolean]' in c for c in calls)
        assert sum('::Void Fill(T)' in c for c in calls) == 1
        assert sum('::Void CopyTo(System.Span`1[T])' in c for c in calls) == 1
        assert not any(i['opcode'] in ['div', 'div.un', 'rem', 'rem.un', 'mul', 'mul.ovf', 'localloc', 'calli'] for i in helper['instructions'])
        report = dict(passed=True, original_core_methods=3189, candidate_core_methods=3190,
            unchanged_core_methods=3188, data_methods=697, helper=HELPER,
            guard_instructions_added=14, original_body_and_edges_exact=True,
            original_locals_preserved_with_one_declared_insertion=True,
            all_existing_flags_exact=True, added_flags=ADDED, public_surface_equal=True,
            new_helper_instructions=len(helper['instructions']))
    assert report is not None
    return report
