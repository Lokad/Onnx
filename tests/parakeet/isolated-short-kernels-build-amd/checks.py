"""Permit only four caller operands and six exact, isolated additions."""
import json

PAIRS = [('PackPanelsB', 'ShortWidePackPanelsB'),
         ('mm_unsafe_vectorized_intrinsics_2x4packed_bump', 'ShortWideMultiply2Rows'),
         ('mm_unsafe_vectorized_intrinsics_3x4packed', 'ShortWideMultiply3Rows'),
         ('mm_unsafe_vectorized_intrinsics', 'ShortWideMultiplyRemainder')]
GENERAL = 'RunFloatMatMulKernel'
DISPATCH = 'RunIsolatedShortWideKernel'
SHORT = 'RunIsolatedShortWidePackedRows'


def one(keys, prefix):
    found = [key for key in keys if key.startswith(prefix)]
    assert len(found) == 1, (prefix, found)
    return found[0]


def operand(key):
    declaring, _, signature = key.split('::', 2)
    return declaring + '::' + signature


def calls_only(before, after, substitutions):
    before, after = json.loads(before), json.loads(after)
    assert {k: v for k, v in before.items() if k != 'instructions'} == {k: v for k, v in after.items() if k != 'instructions'}
    assert len(before['instructions']) == len(after['instructions'])
    counts = {key: 0 for key in substitutions}
    for old, new in zip(before['instructions'], after['instructions'], strict=True):
        if old['opcode'] == 'call' and old['operand'] in substitutions:
            assert new == dict(old, operand=substitutions[old['operand']])
            counts[old['operand']] += 1
        else: assert old == new
    return counts


def normalized_flags(row):
    before = dict(row['method_flags_before']); after = dict(row['method_flags_after'])
    rename = row['compiler_rename']
    if rename:
        assert row['assembly'] == 'Lokad.Onnx.dll'
        assert '<.cctor>b__536_0' in rename['oldKey']
        assert rename['oldKey'].replace('<.cctor>b__536_0', '<.cctor>b__538_0') == rename['newKey']
        assert rename['identical_lambda_body'] == row['normalized_methods'][rename['oldKey']]
        key = 'Lokad.Onnx.Tensor`1[T]::.cctor::Void .cctor()'
        assert rename['beforeConstructor'] == row['normalized_methods'][key]
        expected = rename['beforeConstructor'].replace('b__536_0', 'b__538_0')
        assert expected != rename['beforeConstructor'] and expected == rename['afterConstructor']
        old, new = json.loads(rename['beforeConstructor']), json.loads(rename['afterConstructor'])
        assert {k: v for k, v in old.items() if k != 'instructions'} == {k: v for k, v in new.items() if k != 'instructions'}
        assert len(old['instructions']) == len(new['instructions'])
        changed = [a for a, b in zip(old['instructions'], new['instructions'], strict=True) if a != b]
        assert len(changed) == 1 and changed[0]['opcode'] == 'ldftn'
        assert after[rename['newKey']] == before[rename['oldKey']]
        after[rename['oldKey']] = after.pop(rename['newKey'])
    assert set(before) == set(row['normalized_methods'])
    assert all(after[key] == value for key, value in before.items())
    return before, after


def inventory(value, measured, built, prior):
    assert value['inventory_complete'] and len(value['observations']) == 2 and prior['passed']
    report = None
    for row, (name, count) in zip(value['observations'], [('Lokad.Onnx.dll', 3179), ('Lokad.Onnx.Data.dll', 697)], strict=True):
        assert row['assembly'] == name and row['methods'] == len(row['normalized_methods']) == count
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert row['public_surface_equal'] and not row['removed']
        before, after = normalized_flags(row)
        if name == 'Lokad.Onnx.Data.dll':
            assert row['compiler_rename'] is None and row['unchanged_methods'] == 697
            assert not row['added'] and not row['differences'] and not row['candidate_methods'] and before == after
            continue
        original = row['normalized_methods']; candidate = row['candidate_methods']
        general = one(original, 'Lokad.Onnx.Tensor`1[T]::' + GENERAL + '::')
        dispatch = general.replace(GENERAL, DISPATCH); short = general.replace(GENERAL, SHORT)
        old_target, new_target = operand(general), operand(dispatch)
        callers = [key for key, body in original.items()
                   if any(i['opcode'] == 'call' and i['operand'] == old_target for i in json.loads(body)['instructions'])]
        assert len(callers) == 4 and set(row['differences']) == set(callers)
        assert row['unchanged_methods'] == 3175 and general not in candidate
        assert sum(calls_only(original[key], candidate[key], {old_target: new_target})[old_target] for key in callers) == 4
        copies = {}
        for old_name, new_name in PAIRS:
            keys = [k for k in original if k.startswith('Lokad.Onnx.MathOps::' + old_name + '::') and 'Single*' in k and 'Double*' not in k]
            assert len(keys) == 1
            old_key = keys[0]; new_key = old_key.replace(old_name, new_name)
            assert candidate[new_key] == original[old_key] and after[new_key] == 512
            copies[old_key] = new_key
        added = {dispatch, short, *copies.values()}
        assert set(row['added']) == added and set(candidate) == added | set(callers)
        assert set(after) == set(before) | added and after[dispatch] == 256 and after[short] == 8
        assert original[general] == prior['original_general_body']
        wrapper_map = {prior['old_short_operand']: operand(short), prior['old_general_operand']: old_target}
        assert all(v == 1 for v in calls_only(prior['wrapper_body'], candidate[dispatch], wrapper_map).values())
        helper_map = {operand(old): operand(new) for old, new in copies.items()}
        assert all(v == 1 for v in calls_only(prior['short_body'], candidate[short], helper_map).values())
        report = dict(passed=True, original_core_methods=3179, candidate_core_methods=3185,
            unchanged_core_methods=3175, data_methods=697, callers=callers, call_operands_changed=4,
            original_general_body_exact=True, existing_kernel_bodies_exact=True, existing_implementation_flags_equal=True,
            public_surface_equal=True, copies=copies, exact_copy_bodies=4, dispatcher=dispatch, short_helper=short,
            m43_helper_body_exact_except_declared_calls=True,
            new_flags={k: after[k] for k in sorted(added)}, initializer_rename=row['compiler_rename'] is not None)
    assert report is not None
    return report
