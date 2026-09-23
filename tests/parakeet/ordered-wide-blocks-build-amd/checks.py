"""Allow one exact guarded insertion and three internal methods relative to M54."""
import copy
import json
from il_body import normalized_body

HELPER = 'Lokad.Onnx.Tensor`1[T]::RunIsolatedShortWidePackedRows::Void RunIsolatedShortWidePackedRows(Int32, Int32, Int32, Single*, Single*, Single*, Lokad.Onnx.TensorExecutionOptions)'
TRY = 'Lokad.Onnx.MathOps::TryOrderedWidePackedRows::Boolean TryOrderedWidePackedRows(Int32, Int32, Int32, Single*, Single*, Single*)'
TWO = 'Lokad.Onnx.MathOps::OrderedWideMultiply2Rows::Void OrderedWideMultiply2Rows(Int32, Int32, Int32, Single*, Single*, Single*)'
THREE = TWO.replace('2Rows', '3Rows')
ADDED = {TRY: 256, TWO: 520, THREE: 520}


def operand(key):
    owner, _, signature = key.split('::', 2)
    return owner + '::' + signature


def guarded_body(original):
    """Insert the declared call; preserve the destination of every preexisting edge."""
    before = normalized_body(original)
    ins = before['instructions']
    assert len(ins) == 105 and before['MaxStackSize'] == 8
    assert ins[53] == dict(opcode='brtrue', operand=71)
    assert ins[54] == dict(opcode='ldloc.0', operand='')
    assert ins[55] == dict(opcode='brfalse', operand=64)
    assert ins[71] == dict(opcode='leave', operand=80)
    result = copy.deepcopy(before)
    # The old AVX512-false edge must reach the newly inserted guard at54.
    # Every other continuation at or after54 moves by exactly eight instructions.
    def position(index):
        return index if index < 54 else index + 8
    def edge(index):
        return 54 if index == 54 else position(index)
    for item in result['instructions']:
        op = item['opcode']
        if op != 'break' and op.startswith(('br', 'beq', 'bne', 'bge', 'bgt', 'ble', 'blt', 'leave')):
            item['operand'] = edge(item['operand'])
        else:
            assert op != 'switch'
    for region in result['exceptions']:
        for prefix in ['Try', 'Handler']:
            start = region[prefix + 'Offset']; end = start + region[prefix + 'Length']
            region[prefix + 'Offset'] = position(start)
            region[prefix + 'Length'] = position(end) - position(start)
        assert region['filter'] == -1
    insertion = [dict(opcode=op, operand=arg) for op, arg in [
        ('ldloc.1', ''), ('ldarg.1', ''), ('ldarg.2', ''), ('ldarg.3', ''),
        ('ldloc.3', ''), ('ldarg.s', '05'), ('call', operand(TRY)), ('brtrue', position(71))]]
    result['instructions'][54:54] = insertion
    return result


def new_methods(methods):
    guard = normalized_body(methods[TRY])
    assert not guard['exceptions'] and not guard['locals']
    calls = [i['operand'] for i in guard['instructions'] if i['opcode'] == 'call']
    assert calls == ['System.Runtime.Intrinsics.X86.Fma::Boolean get_IsSupported()', operand(THREE), operand(TWO)]
    assert [i['operand'] for i in guard['instructions'] if i['opcode'] == 'ldc.i4'].count(1024) == 2
    constants = [i['operand'] for i in guard['instructions'] if i['opcode'] == 'ldc.i4']
    assert all(value in constants for value in [64, 32, 67108864])
    assert not any(i['opcode'].startswith(('ldind', 'stind', 'ldobj', 'stobj', 'cpblk', 'initblk')) for i in guard['instructions'])
    kernels = {}
    for key, count in [(TWO, 8), (THREE, 12)]:
        body = normalized_body(methods[key])
        assert not body['exceptions']
        fmas = [i for i in body['instructions'] if i['opcode'] == 'call' and 'Fma::' in i['operand']]
        assert len(fmas) == count and all('MultiplyAdd' in i['operand'] for i in fmas)
        assert sum(i['opcode'] == 'ldc.i4' and i['operand'] == 256 for i in body['instructions']) == 2
        assert sum(i['opcode'] == 'call' and i['operand'] == 'System.Math::Int32 Min(Int32, Int32)' for i in body['instructions']) == 1
        kernels[key] = dict(instructions=len(body['instructions']), fma_calls=count, fixed_reduction_block=256)
    return dict(guard_instructions=len(guard['instructions']), kernels=kernels)


def inventory(value, measured, built, prior):
    assert value['inventory_complete'] and len(value['observations']) == 2 and prior['passed']
    report = None
    for row, (name, count) in zip(value['observations'], [('Lokad.Onnx.dll', 3189), ('Lokad.Onnx.Data.dll', 697)], strict=True):
        assert row['assembly'] == name and row['methods'] == len(row['normalized_methods']) == count
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert row['public_surface_equal'] and row['compiler_rename'] is None and not row['removed']
        before = row['method_flags_before']; after = row['method_flags_after']
        assert set(before) == set(row['normalized_methods'])
        assert all(after.get(key) == flag for key, flag in before.items())
        if name == 'Lokad.Onnx.Data.dll':
            assert row['unchanged_methods'] == 697 and not row['differences'] and not row['added']
            assert not row['candidate_methods'] and before == after
            continue
        assert row['differences'] == [HELPER] and row['unchanged_methods'] == 3188
        assert set(row['added']) == set(ADDED)
        assert set(row['candidate_methods']) == {HELPER, *ADDED}
        assert after == before | ADDED and before[HELPER] == 520
        assert row['normalized_methods'][HELPER] == prior['helper_body'] and prior['helper_key'] == HELPER
        assert normalized_body(row['candidate_methods'][HELPER]) == guarded_body(prior['helper_body'])
        scope = new_methods(row['candidate_methods'])
        report = dict(passed=True, original_core_methods=3189, candidate_core_methods=3192,
                      unchanged_core_methods=3188, data_methods=697, helper=HELPER,
                      helper_instructions_before=105, helper_instructions_after=113,
                      guard_instructions_added=8, original_edges_locals_exceptions_preserved=True,
                      all_existing_flags_exact=True, added_flags=ADDED, public_surface_equal=True,
                      root_entry_and_arithmetic_methods_exact=True, new_methods=scope)
    assert report is not None
    return report
