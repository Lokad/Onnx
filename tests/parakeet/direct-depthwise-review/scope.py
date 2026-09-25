"""Reconcile only the exact compiler names shifted by the four private helpers."""
import copy
import json

RENAMES = [('<>c__DisplayClass547_0', '<>c__DisplayClass543_0'), ('<.cctor>b__548_0', '<.cctor>b__544_0')]


def normalize(value):
    if isinstance(value, str):
        for new, old in RENAMES: value = value.replace(new, old)
        return value
    if isinstance(value, list): return [normalize(v) for v in value]
    if isinstance(value, dict): return {normalize(k):normalize(v) for k,v in value.items()}
    return value


def reconcile(inventory):
    result = copy.deepcopy(inventory); row = result['observations'][0]
    assert row['assembly'] == 'Lokad.Onnx.dll'
    renamed = {k:normalize(k) for k in row['added'] if normalize(k) != k}
    assert len(renamed) == 3 and set(renamed.values()) == set(row['removed'])
    for new, old in renamed.items():
        assert normalize(json.loads(row['candidate_methods'][new])) == json.loads(row['normalized_methods'][old]), new
        assert row['method_flags_after'][new] == row['method_flags_before'][old], new
    references = [k for k in row['differences'] if k.split('::')[1] != 'Conv2DFloatCore']
    assert {k.split('::')[1] for k in references} == {'RunWideProjectionMatMul2DCore', '.cctor'}
    for key in references:
        assert normalize(json.loads(row['candidate_methods'][key])) == json.loads(row['normalized_methods'][key]), key
        assert row['method_flags_after'][key] == row['method_flags_before'][key]
    row['added'] = [k for k in row['added'] if k not in renamed]
    row['removed'] = []
    row['differences'] = [k for k in row['differences'] if k not in references]
    row['candidate_methods'] = {k:v for k,v in row['candidate_methods'].items() if k not in renamed and k not in references}
    row['method_flags_after'] = {renamed.get(k,k):v for k,v in row['method_flags_after'].items()}
    row['unchanged_methods'] += len(renamed)+len(references)
    return result, dict(renamed_methods=renamed, metadata_reference_only=references,
                        all_instructions_locals_exceptions_and_flags_equal=True)
