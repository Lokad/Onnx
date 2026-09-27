"""Bind original full-model consumers to the single diagnosed LSTM candidate."""
import importlib.util
from pathlib import Path
from protocol import pin, read

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
CURRENT = ROOT/'artifacts/parakeet-decoder-packed-row-models-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-decoder-lstm-layout-contracts-v3-amd-20260927'
CONTROL = ROOT/'artifacts/parakeet-decoder-lstm-layout-scalar-baseline-20260927'
SCREEN = ROOT/'artifacts/parakeet-decoder-lstm-layout-timing-amd-20260927'
CAPTURE = ROOT/'artifacts/parakeet-decoder-lstm-runtime-observation-amd-20260927'
DIAGNOSIS = ROOT/'artifacts/parakeet-decoder-lstm-runtime-analysis-20260927'
COMPILED = TOOLS.parent/'decoder-lstm-layout/checks.py'
loader = importlib.util.spec_from_file_location('lstm_compiled_contract', COMPILED)
compiled = importlib.util.module_from_spec(loader); loader.loader.exec_module(compiled)
PROOFS = [
    (CURRENT, 'models', 'closed.json', '5d6832083d103bef9db7bb733b1e96decbae22625f3138680ab6f9a426c78e3b'),
    (QUALIFIED, 'root', 'closed.json', 'd0a78cdd3106d6a72a41303f879bbb2f9ea3bd6a298d778333015f38ccdac246'),
    (CONTRACTS, 'contracts', 'failed.json', '59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'),
    (CONTROL, 'control', 'closed.json', '63c97822999a74921c2e8a3c64e0af9ec6682a6829ba83de52bba353b93b85ec'),
    (SCREEN, 'screen', 'closed.json', 'ac1b5e0e5400d03d84250d6148253208a6d25d3bc9e565a49718721861772fe2'),
    (CAPTURE, 'capture', 'closed.json', 'a00f9195ab9be512b1230357d68e3e1142593014a79680198cc5d517e95b6c85'),
    (DIAGNOSIS, 'diagnosis', 'closed.json', 'dfa4cef574312c04e8d86e6c69151133b2ab4b4f8646bd74fcbe77c12ef23c68')]


def reconcile(root, layout, model_products, selected, candidate):
    assert root['inventory_complete'] and len(root['observations']) == 2
    result = compiled.compiled(layout, selected, candidate)
    for first, second, name, count in zip(root['observations'], layout['observations'],
            ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'], [3284, 697], strict=True):
        assert first['assembly'] == second['assembly'] == name
        assert first['before_sha256'] == model_products[name]['sha256']
        assert first['after_sha256'] == second['before_sha256'] == selected[name]['sha256']
        assert second['after_sha256'] == candidate[name]['sha256']
        assert first['methods'] == first['unchanged_methods'] == second['methods'] == count
        assert not first['differences'] and not first['added'] and not first['removed'] and not first['candidate_methods']
        assert len(first['normalized_methods']) == count and first['normalized_methods'] == second['normalized_methods']
        assert first['method_flags_before'] == first['method_flags_after'] == second['method_flags_before']
        assert first['public_surface_equal'] and second['public_surface_equal']
        assert first['public_surface'] == first['public_surface_after'] == second['public_surface'] == second['public_surface_after']
        assert first['assembly_attributes_before'] == first['assembly_attributes_after'] == second['assembly_attributes_before'] == second['assembly_attributes_after']
        assert set(second['candidate_methods']) == set(second['differences']+second['added'])
        assert set(second['method_flags_after']) == set(second['method_flags_before']) | set(second['added'])
    return dict(passed=True, qualified_model_product=model_products, selected=selected, candidate=candidate,
        original_public_bindings_preserved=True, all_data_methods_exact=True, all_original_method_flags_preserved=True,
        underlying_methods_reconciled=3981, compiled=result, no_consumer_or_product_build=True)


def review():
    inputs = {}; proofs = {}
    for folder, label, filename, digest in PROOFS:
        proof_path = folder/filename
        assert pin(proof_path)['sha256'] == digest
        proof = read(proof_path); proofs[label] = proof
        if folder == CONTRACTS:
            assert not proof['passed'] and proof['evidence_verified'] and proof['terminal']
        else: assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, (label, name)
        inputs[proof_path.relative_to(ROOT).as_posix()] = pin(proof_path)
    assert pin(COMPILED) == read(CONTRACTS/'prepared.json')['files'][COMPILED.relative_to(ROOT).as_posix()]
    inputs[COMPILED.relative_to(ROOT).as_posix()] = pin(COMPILED)
    old, root, control = [read(p/'analysis.json') for p in [CURRENT, QUALIFIED, CONTROL]]
    assert old['passed'] and root['passed'] and not control['contract_regression_found']
    assert not control['original_campaign_passed'] and control['projection_hashes_equal_across_modes']
    selected, candidate = root['built'], control['candidate']
    assert selected == control['baseline'] and control['compiled']['passed']
    inventories = [QUALIFIED/'collected/inventory/instructions.json', CONTRACTS/'collected/inventory/instructions.json']
    for p in inventories: inputs[p.relative_to(ROOT).as_posix()] = pin(p)
    result = reconcile(*map(read, inventories), old['identities']['candidate'], selected, candidate)
    for folder, product in [(QUALIFIED/'collected/runtime', selected), (CONTRACTS/'collected/runtimes/candidate', candidate)]:
        for name, wanted in product.items():
            assert pin(folder/name) == wanted; inputs[(folder/name).relative_to(ROOT).as_posix()] = wanted
    consumers = old['consumers']; assert set(consumers) == {'AudioBenchmark', 'TranscribeReplay'}
    for name, wanted in consumers.items():
        p = CURRENT/'collected/runtimes/candidate'/(name+'.dll')
        assert pin(p) == wanted == proofs['models']['files'][p.relative_to(CURRENT).as_posix()]
        inputs[p.relative_to(ROOT).as_posix()] = wanted
    screen = read(SCREEN/'analysis.json')['performance']; diagnosis = read(DIAGNOSIS/'analysis.json')
    assert not proofs['screen']['admitted'] and not screen['admitted'] and not screen['controls_passed']
    failures = [r for r in screen['controls'] if not r['passed']]
    assert len(failures) == 100 and len(screen['controls']) == 168
    assert len(screen['gates']) == 28 and all(r['passed'] for r in screen['gates'])
    assert diagnosis['passed'] and diagnosis['diagnostic_only'] and not diagnosis['screen_rescored']
    assert diagnosis['capture_closure'] == pin(CAPTURE/'closed.json') and diagnosis['lost'] == 0
    assert {name: diagnosis['product'][name] for name in selected} == selected
    applied = read(QUALIFIED/'bundle/evidence/root-applied.json')
    for name, wanted in applied['source_files'].items(): assert pin(ROOT/name) == wanted, name
    return dict(**result, consumers=consumers, inputs=inputs, component_screen_admitted=False,
        failed_component_controls=failures, runtime_diagnosis=pin(DIAGNOSIS/'closed.json'),
        test_eligibility='PLAN decision after runtime diagnosis: exact full Parakeet correctness, then an independent complete-transcription verdict. Original scalar failure and failed timing screen remain unchanged.')
