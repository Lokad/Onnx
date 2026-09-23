"""Allow exactly the removed upper-row guard relative to the compiled M50 DLLs."""
import copy
import json

DISPATCH = 'RunIsolatedShortWideKernel'


def extended_body(body):
    value = json.loads(body); instructions = value['instructions']
    assert value['InitLocals'] and value['MaxStackSize'] == 7 and not value['locals'] and not value['exceptions']
    assert instructions[:6] == [
        dict(offset=0, opcode='ldarg.0', operand=''),
        dict(offset=1, opcode='ldc.i4.s', operand='30'),
        dict(offset=3, opcode='blt.s', operand='4B'),
        dict(offset=5, opcode='ldarg.0', operand=''),
        dict(offset=6, opcode='ldc.i4.s', operand='40'),
        dict(offset=8, opcode='bge.s', operand='46')]
    assert len(instructions) == 46 and instructions[-1] == dict(offset=95, opcode='ret', operand='')
    result = copy.deepcopy(value)
    result['instructions'] = result['instructions'][:3] + result['instructions'][6:]
    result['instructions'][2]['operand'] = '46'
    for row in result['instructions'][3:]: row['offset'] -= 5
    return result


def inventory(value, measured, built, prior):
    assert value['inventory_complete'] and len(value['observations']) == 2 and prior['passed']
    expected = extended_body(prior['dispatcher_body'])
    for row, (name, count) in zip(value['observations'], [('Lokad.Onnx.dll', 3185), ('Lokad.Onnx.Data.dll', 697)], strict=True):
        assert row['assembly'] == name and row['methods'] == len(row['normalized_methods']) == count
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert row['public_surface_equal'] and not row['removed'] and not row['added'] and row['compiler_rename'] is None
        assert row['method_flags_before'] == row['method_flags_after']
        assert set(row['method_flags_before']) == set(row['normalized_methods'])
        if name == 'Lokad.Onnx.Data.dll':
            assert row['unchanged_methods'] == 697 and not row['differences'] and not row['candidate_methods']
            continue
        key = prior['dispatcher_key']
        assert '::' + DISPATCH + '::' in key
        assert row['normalized_methods'][key] == prior['dispatcher_body']
        assert row['unchanged_methods'] == 3184 and row['differences'] == [key]
        assert set(row['candidate_methods']) == {key}
        assert json.loads(row['candidate_methods'][key]) == expected
    return dict(passed=True, core_methods=3185, unchanged_core_methods=3184, data_methods=697,
        changed_method=prior['dispatcher_key'], removed_upper_row_guard=True, all_other_bodies_exact=True,
        all_flags_exact=True, public_surface_equal=True, no_added_or_removed_methods=True,
        before_instructions=46, after_instructions=43)
