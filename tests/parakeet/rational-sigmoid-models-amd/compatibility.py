"""Bind retained full-model consumers through actual root to the rational candidate."""
from pathlib import Path
from protocol import pin, read

ROOT = Path(__file__).resolve().parents[3]
CURRENT = ROOT/'artifacts/parakeet-pad-current-models-amd-20260926'
QUALIFIED = ROOT/'artifacts/parakeet-pad-current-root-amd-20260926'
BUILD = ROOT/'artifacts/parakeet-rational-sigmoid-build-amd-20260927'
SCREEN = ROOT/'artifacts/parakeet-rational-sigmoid-screen-amd-20260927'
MEMORY = ROOT/'artifacts/parakeet-rational-sigmoid-fallback-diagnostic-amd-20260927'


def reconcile(root, rational, model_products, selected, candidate):
    assert root['inventory_complete'] and rational['inventory_complete']
    assert len(root['observations']) == len(rational['observations']) == 2
    for first, second, name, count in zip(root['observations'], rational['observations'],
            ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'], [3282, 697], strict=True):
        core = name == 'Lokad.Onnx.dll'
        assert first['assembly'] == second['assembly'] == name
        assert first['before_sha256'] == model_products[name]['sha256']
        assert first['after_sha256'] == second['before_sha256'] == selected[name]['sha256']
        assert second['after_sha256'] == candidate[name]['sha256']
        assert first['methods'] == first['unchanged_methods'] == second['methods'] == count
        assert not first['differences'] and not first['added'] and not first['removed'] and not first['candidate_methods']
        assert len(first['normalized_methods']) == count
        assert first['normalized_methods'] == second['normalized_methods']
        assert first['method_flags_before'] == first['method_flags_after'] == second['method_flags_before']
        assert first['public_surface_equal'] and second['public_surface_equal']
        assert first['public_surface'] == first['public_surface_after'] == second['public_surface'] == second['public_surface_after']
        assert first['assembly_attributes_before'] == first['assembly_attributes_after'] == second['assembly_attributes_before'] == second['assembly_attributes_after']
        assert not second['removed'] and second['unchanged_methods'] == count-int(core)
        assert all(second['method_flags_after'][k] == v for k,v in second['method_flags_before'].items())
        if core:
            assert len(second['differences']) == len(second['added']) == 1
            assert '::Sigmoid::' in second['differences'][0] and '::SigmoidRationalVector::' in second['added'][0]
            assert second['method_flags_after'][second['added'][0]] == 8
        else: assert not second['differences'] and not second['added']
        assert set(second['method_flags_after']) == set(second['method_flags_before']) | set(second['added'])
        assert set(second['candidate_methods']) == set(second['differences']+second['added'])
    return dict(passed=True, qualified_model_product=model_products, selected=selected, candidate=candidate,
        original_public_bindings_preserved=True, all_data_methods_exact=True,
        all_original_method_flags_preserved=True, underlying_methods_reconciled=3979,
        changed_core_methods=['Sigmoid'], added_private_methods=['SigmoidRationalVector'],
        no_consumer_or_product_build=True)


def review():
    inputs = {}; proofs = {}
    for folder, digest in [
        (CURRENT, '194eb9a48b64df22d19b51adfc3ebe3accc56004544620123c128c2a76f066bf'),
        (QUALIFIED, '71c80efd687562cba2e2b5d03e9e036b93de09d0a30f74b976b86355ed12fdf0'),
        (BUILD, '3301ae58b42e1fd51435f54191cf4bba42c6cbf8e2ad6bf4279af67ae5d3d89a'),
        (SCREEN, 'fc8a6d9736ff3fd324ebd34ad17a213c857bb178556d4aa7175ee455cc8070b6'),
        (MEMORY, 'c6f0a34eb3467c69829ad958194f5f88375b83abb12fad8450c233bc41586049'),
    ]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']; proofs[folder] = proof
        assert proof['analysis'] == pin(folder/'analysis.json')
        for name in ['closed.json', 'analysis.json']:
            inputs[(folder/name).relative_to(ROOT).as_posix()] = pin(folder/name)
    assert not proofs[SCREEN]['admitted'] and not proofs[MEMORY]['admitted']
    old = read(CURRENT/'analysis.json'); root = read(QUALIFIED/'analysis.json'); build = read(BUILD/'analysis.json')
    inventories = [QUALIFIED/'collected/inventory/instructions.json', BUILD/'build-collected/logs/instructions.json']
    for path, folder in zip(inventories, [QUALIFIED, BUILD], strict=True):
        assert pin(path) == proofs[folder]['files'][path.relative_to(folder).as_posix()]
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    result = reconcile(*map(read, inventories), old['identities']['candidate'], root['built'], build['product'])
    for folder, products in [(QUALIFIED/'collected/runtime', root['built']), (BUILD/'build-collected/runtime', build['product'])]:
        for name, wanted in products.items():
            assert pin(folder/name) == wanted; inputs[(folder/name).relative_to(ROOT).as_posix()] = wanted
    consumers = old['consumers']; assert set(consumers) == {'AudioBenchmark', 'TranscribeReplay'}
    for name, wanted in consumers.items():
        path = CURRENT/'collected/runtimes/candidate'/(name+'.dll')
        assert pin(path) == wanted == proofs[CURRENT]['files'][path.relative_to(CURRENT).as_posix()]
        inputs[path.relative_to(ROOT).as_posix()] = wanted
    screen = read(SCREEN/'analysis.json')
    failures = [c for c in screen['controls'] if not c['passed']]
    assert len(failures) == 13 and len([c for c in screen['rows'] if not c['passed']]) == 4
    assert read(MEMORY/'analysis.json')['diagnostic_only']
    return dict(**result, consumers=consumers, inputs=inputs, component_screen_admitted=False,
        failed_component_controls=failures, failed_component_cases=[c for c in screen['rows'] if not c['passed']],
        test_eligibility='PLAN decision after closed fallback diagnosis: correctness only; no failed verdict changed.')
