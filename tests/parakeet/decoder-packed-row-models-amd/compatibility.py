"""Bind unchanged model consumers through the actual root to the packed-row pair."""
import importlib.util
from pathlib import Path
from protocol import pin, read

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
CURRENT = ROOT/'artifacts/parakeet-rational-sigmoid-models-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-rational-sigmoid-root-amd-20260927'
FIRST = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-amd-20260927'
INVENTORY = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v2-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v3-amd-20260927'
SCREEN = ROOT/'artifacts/parakeet-decoder-packed-row-screen-v2-amd-20260927'
DIAGNOSIS = ROOT/'artifacts/parakeet-decoder-unmapped-calls-amd-20260927'
COMPILED = TOOLS.parent/'decoder-packed-row/compiled.py'
loader = importlib.util.spec_from_file_location('fixed_row_compiled_contract', COMPILED)
compiled = importlib.util.module_from_spec(loader); loader.loader.exec_module(compiled)


def reconcile(root, packed, model_products, selected, candidate):
    assert root['inventory_complete'] and len(root['observations']) == 2
    result = compiled.review(packed, {'products': {'candidate': candidate['Lokad.Onnx.dll']}}, selected)
    for first, second, name, count in zip(root['observations'], packed['observations'],
            ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'], [3283, 697], strict=True):
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
        if name == 'Lokad.Onnx.dll': assert second['method_flags_after'][second['added'][0]] == 0
    return dict(passed=True, qualified_model_product=model_products, selected=selected, candidate=candidate,
        original_public_bindings_preserved=True, all_data_methods_exact=True, all_original_method_flags_preserved=True,
        underlying_methods_reconciled=3980, compiled=result, no_consumer_or_product_build=True)


def review():
    inputs = {}; proofs = {}
    for folder, filename, digest in [
        (CURRENT, 'closed.json', '3d2e01b5aca3f66c434d5d1e8124c2b0d08327bc14dd37409c0e1884619769d8'),
        (QUALIFIED, 'closed.json', 'f6df53ce2ab773898bc84f144abeda97908d383fef9a4df4b7299a22a4d3594d'),
        (INVENTORY, 'failed.json', '2c74d4aede20993b14a988c8a04399f238954f0c6455c78395e1cb1cda581478'),
        (CONTRACTS, 'closed.json', 'fc00688c8ef0e65e5d5109f1808209add023e740aabc5b4312647e547df7d61d'),
        (SCREEN, 'closed.json', '3e6a7562db1946954f2cddba10b2ee31958854e66c50fbb55848ed04e3e75148'),
        (DIAGNOSIS, 'closed.json', '966941d1e040211b840259438fe8bb786aea23dc3c8f9987208cf242b5a7e986')]:
        assert pin(folder/filename)['sha256'] == digest
        proof = read(folder/filename); proofs[folder] = proof
        assert proof.get('passed') or (folder == INVENTORY and proof['evidence_verified'] and proof['terminal'])
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
        inputs[(folder/filename).relative_to(ROOT).as_posix()] = pin(folder/filename)
    assert pin(COMPILED) == read(CONTRACTS/'prepared.json')['files'][COMPILED.relative_to(ROOT).as_posix()]
    inputs[COMPILED.relative_to(ROOT).as_posix()] = pin(COMPILED)
    old = read(CURRENT/'analysis.json'); root = read(QUALIFIED/'analysis.json'); contracts = read(CONTRACTS/'analysis.json')
    assert old['passed'] and root['passed'] and contracts['passed'] and contracts['compiled']['passed']
    selected = root['built']
    candidate = dict(selected, **{'Lokad.Onnx.dll': contracts['products']['candidate']})
    assert contracts['products']['current'] == selected['Lokad.Onnx.dll']
    inventories = [QUALIFIED/'collected/inventory/instructions.json', INVENTORY/'collected/inventory/instructions.json']
    for p in inventories: inputs[p.relative_to(ROOT).as_posix()] = pin(p)
    result = reconcile(*map(read, inventories), old['identities']['candidate'], selected, candidate)
    for folder, product in [(QUALIFIED/'collected/runtime', selected), (FIRST/'collected/runtimes/candidate', candidate)]:
        for name, wanted in product.items():
            assert pin(folder/name) == wanted; inputs[(folder/name).relative_to(ROOT).as_posix()] = wanted
    consumers = old['consumers']; assert set(consumers) == {'AudioBenchmark', 'TranscribeReplay'}
    for name, wanted in consumers.items():
        p = CURRENT/'collected/runtimes/candidate'/(name+'.dll')
        assert pin(p) == wanted == proofs[CURRENT]['files'][p.relative_to(CURRENT).as_posix()]
        inputs[p.relative_to(ROOT).as_posix()] = wanted
    screen = read(SCREEN/'analysis.json'); diagnosis = read(DIAGNOSIS/'analysis.json')
    assert not proofs[SCREEN]['admitted'] and not screen['admitted']
    assert not proofs[DIAGNOSIS]['admitted'] and not diagnosis['first_call_explanation_supported']
    failures = [r for r in screen['controls'] if not r['passed']]
    failed_cases = [r for r in screen['rows'] if not r['passed']]
    assert len(failures) == len(failed_cases) == 2
    applied = read(QUALIFIED/'bundle/evidence/root-applied.json')
    for name, wanted in applied['source_files'].items(): assert pin(ROOT/name) == wanted, name
    return dict(**result, consumers=consumers, inputs=inputs, component_screen_admitted=False,
        failed_component_controls=failures, failed_component_cases=failed_cases,
        failed_first_call_prediction=True,
        test_eligibility='PLAN decision after closed per-call diagnosis: exact full-model correctness; all failed verdicts remain.')
