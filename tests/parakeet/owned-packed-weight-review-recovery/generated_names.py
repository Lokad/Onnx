"""Prove the exact +5 compiler ordinal shift without ignoring any IL instruction."""
import copy
import json
import re

ORDINALS = {492, 493, 494, 495, 496, 516, 517, 520, 524, 525, 526, 539, 540}
PATTERN = re.compile(r'(<>c__DisplayClass|<>9__|>b__|>g__[^|<>]+\|)(\d+)(?=_)')
MEMBER_OPS = {'call', 'callvirt', 'newobj', 'ldftn', 'ldvirtftn', 'ldfld', 'ldflda',
              'stfld', 'ldsfld', 'ldsflda', 'stsfld', 'ldtoken', 'castclass',
              'isinst', 'initobj', 'box', 'unbox', 'unbox.any', 'sizeof', 'constrained.'}


def renamed(value):
    # Only references into this partial generic class changed ordinal.
    if not value.startswith('Lokad.Onnx.Tensor`1'):
        return value
    def replace(match):
        number = int(match[2])
        return match[1] + str(number + 5 if number in ORDINALS else number)
    return PATTERN.sub(replace, value)


def body_after_rename(value):
    if value == 'NO-BODY':
        return value
    body = json.loads(value)
    for local in body['locals']:
        local['type'] = renamed(local['type'])
    for clause in body['exceptions']:
        if clause['caught'] is not None:
            clause['caught'] = renamed(clause['caught'])
    for instruction in body['instructions']:
        if instruction['opcode'] in MEMBER_OPS:
            instruction['operand'] = renamed(instruction['operand'])
    # Literal strings, branch offsets, numbers, stack size and all opcodes survive.
    return body


def reconcile(row):
    if row['assembly'] != 'Lokad.Onnx.dll':
        return dict(row, compiler_renames={})
    old = row['normalized_methods']
    candidate = {k: v for k, v in old.items() if k not in row['removed']}
    candidate.update(row['candidate_methods'])
    assert set(candidate) == set(row['method_flags_after'])
    mapping = {key: renamed(key) for key in old}
    assert len(set(mapping.values())) == len(mapping), 'Rename must be injective'
    assert set(mapping.values()) <= set(candidate), 'Every old method must survive'
    differences = []
    for key, target in mapping.items():
        actual = candidate[target]
        if actual != 'NO-BODY':
            actual = json.loads(actual)
        if body_after_rename(old[key]) != actual:
            differences.append(key)
        assert row['method_flags_before'][key] == row['method_flags_after'][target], key
    added = sorted(set(candidate) - set(mapping.values()))
    result = copy.deepcopy(row)
    result.update(removed=[], added=added, differences=differences,
                  unchanged_methods=len(old) - len(differences),
                  compiler_renames={k: v for k, v in mapping.items() if k != v})
    result['method_flags_after'] = {k: row['method_flags_after'][v] for k, v in mapping.items()}
    result['method_flags_after'].update({k: row['method_flags_after'][k] for k in added})
    result['candidate_methods'] = {k: candidate[mapping[k]] for k in differences}
    result['candidate_methods'].update({k: candidate[k] for k in added})
    assert len(result['compiler_renames']) == 41
    return result
