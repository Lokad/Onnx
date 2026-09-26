"""Prove that the retained observer applies to the fully qualified padding root."""
from run import ROOT, TOOLS, PARENT, ORIGINAL, pin, read, load

OLD = ROOT / 'artifacts/parakeet-packed-final-row-profile-amd-20260925'
BUILD = ROOT / 'artifacts/parakeet-pad-current-build-amd-20260926'
QUALIFIED_ROOT = ROOT / 'artifacts/parakeet-pad-current-root-amd-20260926'
BASELINE_ROOT = ROOT / 'artifacts/parakeet-owned-batch-isolation-root-policy-amd-20260925'
CANDIDATE_CORE = 'a74acb17524f23be13e81ade871b2b2ffea2afde5bcdd12e339b75e0197edf10'
CANDIDATE_DATA = 'be954dc40376400336167f3153fcf5e388bf6bb0df14af595420a0aa30471b5f'


def metadata_and_data(root_inventory, baseline_inventory, candidate_inventory, measured, built):
    assert root_inventory['inventory_complete'] and baseline_inventory['inventory_complete']
    assert candidate_inventory['inventory_complete']
    core, data = root_inventory['observations']
    old_core, old_data = baseline_inventory['observations']
    _, candidate_data = candidate_inventory['observations']
    assert measured['Lokad.Onnx.dll']['sha256'] == CANDIDATE_CORE
    assert measured['Lokad.Onnx.Data.dll']['sha256'] == CANDIDATE_DATA
    for row, old, name, count in [(core, old_core, 'Lokad.Onnx.dll', 3282),
                                  (data, old_data, 'Lokad.Onnx.Data.dll', 697)]:
        assert row['assembly'] == old['assembly'] == name
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert row['methods'] == row['unchanged_methods'] == len(row['normalized_methods']) == count
        assert not any(row[k] for k in ['differences', 'added', 'removed', 'candidate_methods'])
        assert row['method_flags_before'] == row['method_flags_after']
        assert len(row['method_flags_after']) == count
        assert row['public_surface_equal']
        assert row['public_surface'] == row['public_surface_after'] == old['public_surface_after']
        assert row['assembly_attributes_before'] == row['assembly_attributes_after'] == old['assembly_attributes_after']
    assert candidate_data['before_sha256'] == old_data['after_sha256']
    assert candidate_data['after_sha256'] == measured['Lokad.Onnx.Data.dll']['sha256']
    assert candidate_data['methods'] == candidate_data['unchanged_methods'] == 697
    assert not any(candidate_data[k] for k in ['differences', 'added', 'removed', 'candidate_methods'])
    assert data['normalized_methods'] == candidate_data['normalized_methods'] == old_data['normalized_methods']
    assert data['method_flags_after'] == candidate_data['method_flags_after'] == candidate_data['method_flags_before'] == old_data['method_flags_after']
    return dict(passed=True, data_methods_equal=697, data_flags_equal=True,
                public_surfaces_equal=True, assembly_attributes_equal=True,
                actual_root_inventory_required=True)


def review_observer():
    parent = load('retained_parent_observer_scope', PARENT / 'reuse.py').review_observer()
    inputs = dict(parent['inputs'])

    def retain(path):
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
        return read(path)

    assert pin(BUILD / 'closed.json')['sha256'] == '1835c79eda505c056cf796702ca734b6e43c65e066b3e54bb566ac25885c4018'
    for base in [BUILD, QUALIFIED_ROOT]:
        proof = retain(base / 'closed.json')
        assert proof['passed'] and proof['analysis'] == pin(base / 'analysis.json')
        for name, wanted in proof['files'].items():
            assert pin(base / name) == wanted, name
    candidate = retain(BUILD / 'analysis.json')
    root = retain(QUALIFIED_ROOT / 'analysis.json')
    assert root['root_source_verified'] and root['measured'] == candidate['built']
    assert candidate['measured'] == parent['product']
    assert root['inventory'] == dict(passed=True, core_methods=3282, data_methods=697,
        public_surface_equal=True, assembly_attributes_equal=True,
        method_bodies_equal=True, implementation_flags_equal=True)
    current = retain(QUALIFIED_ROOT / 'collected/inventory/instructions.json')
    baseline = retain(BASELINE_ROOT / 'collected/inventory/instructions.json')
    intermediate = retain(BUILD / 'collected/inventory/instructions.json')
    scope = metadata_and_data(current, baseline, intermediate, root['measured'], root['built'])
    for name, wanted in root['built'].items():
        path = QUALIFIED_ROOT / 'collected/runtime' / name
        assert pin(path) == wanted
        inputs[path.relative_to(ROOT).as_posix()] = wanted
    for path in [PARENT / 'reuse.py', PARENT / 'run.py', ORIGINAL / 'compiled_scope.py', TOOLS / 'reuse.py']:
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    return dict(passed=True, observer_rebuilt=False, product=root['built'],
                reference_product=root['measured'], parent_product=parent['product'],
                root_closure=pin(QUALIFIED_ROOT / 'closed.json'),
                root_inventory=pin(QUALIFIED_ROOT / 'collected/inventory/instructions.json'),
                root_metadata_equal=True, data_compiled_scope_equal=True, scope=scope,
                original_review=parent['original_review'], original_inventory=parent['original_inventory'],
                methods=parent['methods'], consumer=parent['consumer'], observed_data=parent['observed_data'],
                inputs=inputs)


if __name__ == '__main__':
    import json
    result = review_observer()
    print(json.dumps({k: v for k, v in result.items() if k not in ['inputs', 'methods']}))
