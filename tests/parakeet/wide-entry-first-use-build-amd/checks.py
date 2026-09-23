"""Prove four caller substitutions, an exact entry clone, and one helper flag."""
import json
import re

ENTRY = 'MatMul2DCore'
DISPATCH = 'DispatchWideProjectionMatMul2DCore'
CLONE = 'RunWideProjectionMatMul2DCore'
HELPER = 'RunIsolatedShortWidePackedRows'


def one(keys, prefix):
    found = [key for key in keys if key.startswith(prefix)]
    assert len(found) == 1, (prefix, found)
    return found[0]


def operand(key):
    declaring, _, signature = key.split('::', 2)
    return declaring + '::' + signature


def calls_only(before, after, old_target, new_target):
    before, after = json.loads(before), json.loads(after)
    assert {k:v for k,v in before.items() if k != 'instructions'} == {k:v for k,v in after.items() if k != 'instructions'}
    assert len(before['instructions']) == len(after['instructions'])
    count = 0
    for old, new in zip(before['instructions'], after['instructions'], strict=True):
        if old['opcode'] == 'call' and old['operand'] == old_target:
            assert new == dict(old, operand=new_target); count += 1
        else: assert old == new
    assert count == 1


def replace_body(body, substitutions):
    def change(value):
        if isinstance(value, str):
            for old, new in substitutions.items(): value = value.replace(old, new)
            return value
        if isinstance(value, list): return [change(item) for item in value]
        if isinstance(value, dict): return {key:change(item) for key,item in value.items()}
        return value
    return change(json.loads(body))


def normalized_flags(row):
    before = dict(row['method_flags_before']); after = dict(row['method_flags_after'])
    rename = row['compiler_rename']
    if rename:
        assert row['assembly'] == 'Lokad.Onnx.dll'
        old = re.search(r'<\.cctor>b__\d+_0', rename['oldKey']).group()
        new = re.search(r'<\.cctor>b__\d+_0', rename['newKey']).group()
        assert old != new and rename['oldKey'].replace(old, new) == rename['newKey']
        assert rename['identical_lambda_body'] == row['normalized_methods'][rename['oldKey']]
        ctor = 'Lokad.Onnx.Tensor`1[T]::.cctor::Void .cctor()'
        assert rename['beforeConstructor'] == row['normalized_methods'][ctor]
        first, second = json.loads(rename['beforeConstructor']), json.loads(rename['afterConstructor'])
        assert replace_body(rename['beforeConstructor'], {old:new}) == second
        changes = [(a,b) for a,b in zip(first['instructions'], second['instructions'], strict=True) if a != b]
        assert len(changes) == 1 and changes[0][0]['opcode'] == 'ldftn'
        assert changes[0][1] == dict(changes[0][0],operand=changes[0][0]['operand'].replace(old,new))
        assert after[rename['newKey']] == before[rename['oldKey']]
        after[rename['oldKey']] = after.pop(rename['newKey'])
    assert set(before) == set(row['normalized_methods'])
    return before, after


def inventory(value, measured, built, prior):
    assert value['inventory_complete'] and len(value['observations']) == 2 and prior['passed']
    report = None
    for row, (name,count) in zip(value['observations'], [('Lokad.Onnx.dll',3185),('Lokad.Onnx.Data.dll',697)], strict=True):
        assert row['assembly'] == name and row['methods'] == len(row['normalized_methods']) == count
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert row['public_surface_equal'] and not row['removed']
        before, after = normalized_flags(row)
        if name == 'Lokad.Onnx.Data.dll':
            assert row['compiler_rename'] is None and row['unchanged_methods'] == 697
            assert not row['added'] and not row['differences'] and not row['candidate_methods'] and before == after
            continue
        original = row['normalized_methods']; candidate = row['candidate_methods']
        entry = one(original, 'Lokad.Onnx.Tensor`1[T]::'+ENTRY+'::')
        helper = one(original, 'Lokad.Onnx.Tensor`1[T]::'+HELPER+'::')
        dispatch = entry.replace(ENTRY,DISPATCH); clone = entry.replace(ENTRY,CLONE)
        old_target, new_target = operand(entry), operand(dispatch)
        callers = [key for key, body in original.items() if body != 'NO-BODY'
                   and any(i['opcode'] == 'call' and i['operand'] == old_target for i in json.loads(body)['instructions'])]
        assert len(callers) == 4 and set(row['differences']) == set(callers) and row['unchanged_methods'] == 3181
        for key in callers: calls_only(original[key],candidate[key],old_target,new_target)
        old_closure = json.loads(original[entry])['locals'][0]['type']
        new_closure = json.loads(candidate[clone])['locals'][0]['type']
        assert re.fullmatch(r'Lokad.Onnx.Tensor`1\+<>c__DisplayClass\d+_0\[T\]', old_closure)
        assert re.fullmatch(r'Lokad.Onnx.Tensor`1\+<>c__DisplayClass\d+_0\[T\]', new_closure)
        assert old_closure != new_closure
        old_lambda = one(original,old_closure+'::<'+ENTRY+'>b__0::')
        new_lambda = old_lambda.replace(old_closure,new_closure).replace(ENTRY,CLONE)
        old_ctor = one(original,old_closure+'::.ctor::')
        new_ctor = old_ctor.replace(old_closure,new_closure)
        substitutions = {old_closure:new_closure,'<'+ENTRY+'>b__0':'<'+CLONE+'>b__0'}
        for old,new in [(entry,clone),(old_lambda,new_lambda),(old_ctor,new_ctor)]:
            assert replace_body(original[old],substitutions) == json.loads(candidate[new]),new
        added = {dispatch,clone,new_lambda,new_ctor}
        assert set(row['added']) == added and set(candidate) == added | set(callers)
        assert set(after) == set(before) | added
        assert before[helper] == 8 and after[helper] == 520
        assert all(after[k] == v for k,v in before.items() if k != helper)
        assert after[dispatch] == 256 and after[clone] == 520
        assert after[new_lambda] == before[old_lambda] and after[new_ctor] == before[old_ctor]
        assert original[entry] == prior['entry_body'] and entry == prior['entry_key']
        wrapper = json.loads(candidate[dispatch])
        targets = [i['operand'] for i in wrapper['instructions'] if i['opcode'] == 'call']
        assert targets.count(operand(clone)) == targets.count(operand(entry)) == 1
        assert not wrapper['exceptions']
        constants = [(i['opcode'],i['operand']) for i in wrapper['instructions'] if i['opcode'].startswith('ldc.')]
        assert ('ldc.i4.s','30') in constants and constants.count(('ldc.i4','00040000')) == 2
        assert ('ldc.i4','00000004') in constants
        report = dict(passed=True,original_core_methods=3185,candidate_core_methods=3189,data_methods=697,
            unchanged_core_bodies=3181,callers=callers,call_operands_changed=4,entry_original_exact=True,
            clone=clone,dispatcher=dispatch,helper=helper,clone_closure=new_closure,clone_lambda=new_lambda,
            exact_cloned_bodies=3,existing_flag_changes={helper:[8,520]},shared_flags_exact=True,
            new_flags={k:after[k] for k in sorted(added)},public_surface_equal=True,
            initializer_rename=row['compiler_rename'] is not None)
    assert report is not None
    return report
