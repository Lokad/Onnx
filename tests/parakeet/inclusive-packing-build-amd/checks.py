"""Admit exactly one signed branch opcode change, preserving the whole method."""
import json

HELPER = 'Lokad.Onnx.GraphPacking::FitsPackBudget::Boolean FitsPackBudget(Int32, Int32)'


def expected_body(original):
    body = json.loads(original)
    instructions = body['instructions']
    assert len(instructions) == 22
    assert instructions[7] == dict(offset=9, opcode='ldc.i4', operand='00100000')
    assert instructions[8] == dict(offset=14, opcode='bge.s', operand='11')
    # Same encoded width and branch destination: only equality is now admitted.
    instructions[8]['opcode'] = 'bgt.s'
    return body


def inventory(value, measured, built, prior):
    assert value['inventory_complete'] and len(value['observations']) == 2 and prior['passed']
    for row, (name, count) in zip(value['observations'], [('Lokad.Onnx.dll', 3189), ('Lokad.Onnx.Data.dll', 697)], strict=True):
        assert row['assembly'] == name and row['methods'] == len(row['normalized_methods']) == count
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert row['public_surface_equal'] and row['compiler_rename'] is None
        assert not row['added'] and not row['removed']
        assert row['method_flags_before'] == row['method_flags_after']
        assert set(row['method_flags_before']) == set(row['normalized_methods'])
        if name == 'Lokad.Onnx.Data.dll':
            assert row['unchanged_methods'] == 697 and not row['differences'] and not row['candidate_methods']
            continue
        assert row['differences'] == [HELPER] and row['unchanged_methods'] == 3188
        assert set(row['candidate_methods']) == {HELPER}
        assert prior['helper_key'] == HELPER and row['normalized_methods'][HELPER] == prior['helper_body']
        assert json.loads(row['candidate_methods'][HELPER]) == expected_body(prior['helper_body'])
    return dict(passed=True, core_methods=3189, unchanged_core_methods=3188, data_methods=697,
                changed_method=HELPER, instructions=22, changed_instruction_offset=14,
                opcode_before='bge.s', opcode_after='bgt.s', branch_operand='11',
                other_instructions_headers_locals_exceptions_exact=True,
                implementation_flags_equal=True, public_surface_equal=True,
                budgets_and_kernels_unchanged=True)
