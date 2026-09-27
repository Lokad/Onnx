"""Bind the original model consumers to the single pointwise remainder change."""
from pathlib import Path
from protocol import pin, read

ROOT = Path(__file__).resolve().parents[3]
CURRENT = ROOT/'artifacts/parakeet-decoder-lstm-layout-models-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-lstm-layout-root-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-pointwise-tail-contracts-amd-20260927'
ARITHMETIC = ROOT/'artifacts/parakeet-pointwise-tail-arithmetic-contracts-amd-20260927'
SCREEN = ROOT/'artifacts/parakeet-pointwise-tail-timing-amd-20260927'
CAPTURE = ROOT/'artifacts/parakeet-pointwise-tail-runtime-observation-amd-20260927'
DIAGNOSIS = ROOT/'artifacts/parakeet-pointwise-tail-runtime-analysis-20260927'
PROOFS = [
    (CURRENT,'models','closed.json','1acb1d884eb5293e0487e1b34d8c842b7240823040e63d6dc2a043ec89f7a111'),
    (QUALIFIED,'root','closed.json','efb99eea455647c64bbe0811c1b7d46387add59049ac81e48a926b250e7c42da'),
    (CONTRACTS,'contracts','closed.json','58cb40269af7cda8252007bc059f04ccd4e59697d258e3affbbe48329b8208d9'),
    (ARITHMETIC,'arithmetic','closed.json','558f2a523febd0d794bc7da6cdae3381513b7ff3d75dfd4ca4f133f618821cc8'),
    (SCREEN,'screen','closed.json','b9ff0d6050068ad8ad9ed9fa7a50350bb7e99079e10316f077183bea8ef0765c'),
    (CAPTURE,'capture','closed.json','473104209f3a9161dc0600df123581fbb6fe626bea8894be69388b953277332e'),
    (DIAGNOSIS,'diagnosis','closed.json','75f3b25985a8ff4efdc7465de0444b3e667deb0c5922755b0da8a8a5770601c4')]


def reconcile(root, tail, model_products, selected, candidate):
    assert root['inventory_complete'] and tail['inventory_complete']
    assert len(root['observations']) == len(tail['observations']) == 2
    scope = []
    for first, second, name, count in zip(root['observations'], tail['observations'],
            ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll'], [3286,697], strict=True):
        assert first['assembly'] == second['assembly'] == name
        assert first['before_sha256'] == model_products[name]['sha256']
        assert first['after_sha256'] == second['before_sha256'] == selected[name]['sha256']
        assert second['after_sha256'] == candidate[name]['sha256']
        assert first['methods'] == first['unchanged_methods'] == second['methods'] == count
        assert not first['differences'] and not first['added'] and not first['removed'] and not first['candidate_methods']
        assert len(first['normalized_methods']) == count and first['normalized_methods'] == second['normalized_methods']
        assert first['method_flags_before'] == first['method_flags_after'] == second['method_flags_before']
        assert not second['removed']
        if name == 'Lokad.Onnx.dll':
            assert len(second['differences']) == 1 and {n.split('::')[1] for n in second['differences']} == {'mm_unsafe_vectorized_intrinsics_2x4packed_bump'}
            assert len(second['added']) == 2 and {n.split('::')[1] for n in second['added']} == {'PackedColumnTailEightRows','PackedColumnMaskedEightRows'}
            assert all(n.startswith('Lokad.Onnx.MathOps::') for n in second['differences']+second['added'])
            assert second['unchanged_methods'] == count-1
        else:
            assert not second['differences'] and not second['added'] and second['unchanged_methods'] == count
        for row in [first, second]:
            assert row['public_surface_equal'] and row['public_surface'] == row['public_surface_after']
            assert row['assembly_attributes_before'] == row['assembly_attributes_after']
        assert first['public_surface_after'] == second['public_surface']
        assert first['assembly_attributes_after'] == second['assembly_attributes_before']
        assert all(second['method_flags_after'][n] == f for n,f in second['method_flags_before'].items())
        assert set(second['method_flags_after']) == set(second['method_flags_before']) | set(second['added'])
        assert set(second['candidate_methods']) == set(second['differences']+second['added'])
        scope.append(dict(assembly=name, unchanged=second['unchanged_methods'], changed=second['differences'], added=second['added']))
    return dict(passed=True, selected=selected, candidate=candidate,
        qualified_model_product=model_products, underlying_methods_reconciled=3983, compiled_scope=scope,
        original_public_bindings_preserved=True, all_data_methods_exact=True,
        all_original_method_flags_preserved=True, no_consumer_or_product_build=True)


def review():
    inputs = {}; proofs = {}
    for folder, label, filename, digest in PROOFS:
        path = folder/filename; assert pin(path)['sha256'] == digest
        proof = read(path); proofs[label] = proof
        for name,wanted in proof['files'].items(): assert pin(folder/name) == wanted, (label,name)
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    assert proofs['models']['passed'] and proofs['root']['passed']
    assert proofs['contracts']['completed'] and not proofs['contracts']['passed']
    assert proofs['arithmetic']['completed'] and proofs['arithmetic']['arithmetic_contract_passed']
    assert proofs['screen']['completed'] and not proofs['screen']['component_admitted']
    assert proofs['capture']['passed'] and proofs['diagnosis']['passed']
    build = CONTRACTS/'build-review.json'
    assert pin(build)['sha256'] == '51090680e6b172287122ef15c5f7e5a3ae2eaa41f083a76cdcb314f418ddc227'
    built = read(build); assert built['passed']
    selected, candidate = built['products']['baseline'], built['products']['candidate']
    assert read(QUALIFIED/'analysis.json')['built'] == selected
    arithmetic = read(ARITHMETIC/'analysis.json')
    assert arithmetic['arithmetic_contract_passed'] and arithmetic['products'] == built['products']
    codegen = ARITHMETIC/'codegen-review.json'
    assert pin(codegen)['sha256'] == '3715408c95a3164b2f06e36db4aeb2bea95f08ee8f7afb03cbb38e9747d9d692'
    assert read(codegen)['passed'] and read(codegen)['numerical_closure'] == pin(ARITHMETIC/'closed.json')
    inventories = [QUALIFIED/'collected/inventory/instructions.json', CONTRACTS/'build-collected/logs/instructions.json']
    model = read(CURRENT/'analysis.json')
    result = reconcile(*map(read,inventories), model['identities']['candidate'], selected, candidate)
    assert result['compiled_scope'] == built['scope']
    for p in [build,codegen,*inventories]: inputs[p.relative_to(ROOT).as_posix()] = pin(p)
    for role,product in [('baseline',selected),('candidate',candidate)]:
        for name,wanted in product.items():
            p = CONTRACTS/'build-collected/runtime'/role/name
            assert pin(p) == wanted; inputs[p.relative_to(ROOT).as_posix()] = wanted
    consumers = model['consumers']; assert set(consumers) == {'AudioBenchmark','TranscribeReplay'}
    for name,wanted in consumers.items():
        p = CURRENT/'collected/runtimes/candidate'/(name+'.dll')
        assert pin(p) == wanted; inputs[p.relative_to(ROOT).as_posix()] = wanted
    diagnosis = read(DIAGNOSIS/'analysis.json')
    assert diagnosis['passed'] and diagnosis['diagnostic_only'] and not diagnosis['screen_rescored']
    assert diagnosis['capture_closure'] == pin(CAPTURE/'closed.json') and diagnosis['product'] == candidate
    assert diagnosis['original_calls'] == 400 and diagnosis['lost'] == 0
    performance = read(SCREEN/'analysis.json')['performance']
    failures = [r for r in performance['controls'] if not r['passed']]
    assert len(failures) == 38 and len(performance['controls']) == 246
    assert sum(not r['passed'] for r in performance['gates']) == 1
    root_files = {n.removeprefix('source/'):w for n,w in read(QUALIFIED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(root_files) == 443
    for name,wanted in root_files.items(): assert pin(ROOT/name) == wanted, name
    inputs.update(root_files)
    return dict(**result, consumers=consumers, inputs=inputs, component_screen_admitted=False,
        failed_component_controls=failures, runtime_diagnosis=pin(DIAGNOSIS/'closed.json'),
        arithmetic_qualification=pin(ARITHMETIC/'closed.json'), codegen_review=pin(codegen),
        test_eligibility='Unchanged numerically qualified candidate after runtime diagnosis; full-model correctness, then independent complete-application judgment. Failed component screen remains unchanged.')
