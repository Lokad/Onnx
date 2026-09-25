"""Bind the reviewed observer to unchanged Data and the qualified current Core."""
from run import ROOT, ORIGINAL, pin, read, load

OLD = ROOT/'artifacts/parakeet-packed-final-row-profile-amd-20260925'
CONTRACTS = ROOT/'artifacts/parakeet-owned-batch-isolation-build-amd-20260925'
DEPTHWISE = ROOT/'artifacts/parakeet-direct-depthwise-build-v2-amd-20260925'
CORE = 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
DATA = '01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'
OBSERVER = '51b6d2bb72640a99d8ed4e97334525dde8be60ac1ecb3c25ea690956bfcab2a2'
RUNNER = '38ab5c7e65aa00ace50e8704348830286047e0c96ebc0a04b66c6e54d063899c'


def review_observer():
    inputs = {}
    def retain(path):
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
        return read(path)
    review = retain(OLD/'build-review.json')
    assert pin(OLD/'build-review.json')['sha256'] == '1f6ff9bb6fe68525dad1a49fc1273406cff86a05d6e5f44a2256e040f0b9a04a'
    assert review['passed'] and review['warnings'] == []
    folder = OLD/'build-collected'
    for field, name in [('built','built.json'),('inventory','inventory/instructions.json'),('collection','build-collection.json')]:
        assert review[field] == pin(folder/name)
    receipt = retain(folder/'build-collection.json')
    assert receipt['terminal'] and receipt['code'] == 0
    for name, wanted in receipt['files'].items():
        assert pin(folder/name) == wanted, name
    built = retain(folder/'built.json')
    for name, wanted in built['runtime_files'].items():
        assert pin(folder/name) == wanted, name
        inputs[(folder/name).relative_to(ROOT).as_posix()] = wanted
    assert built['data']['sha256'] == OBSERVER and built['consumer']['sha256'] == RUNNER
    assert pin(folder/'runtime-control/Lokad.Onnx.Data.dll')['sha256'] == DATA
    inventory = retain(folder/'inventory/instructions.json')
    reference = retain(OLD/'bundle/evidence/original-observer-instructions.json')
    assert inventory['inventory_complete']
    runner, data = inventory['observations']
    reference_data, = [r for r in reference['observations'] if r['assembly'] == 'Lokad.Onnx.Data.dll']
    checker = load('retained_observer_compiled_scope', ORIGINAL/'compiled_scope.py')
    inputs[(ORIGINAL/'compiled_scope.py').relative_to(ROOT).as_posix()] = pin(ORIGINAL/'compiled_scope.py')
    methods = [checker.verify_runner(runner, RUNNER), checker.verify_data(data, DATA, OBSERVER, reference_data)]
    assert methods == review['methods']
    inventories = []
    for base in [DEPTHWISE, CONTRACTS]:
        proof = retain(base/'closed.json')
        qualified = retain(base/'build-review.json')
        path = base/'build-collected/logs/instructions.json'
        value = retain(path)
        assert proof['passed'] and qualified['passed'] and qualified['inventory'] == pin(path)
        assert proof['files']['build-review.json'] == pin(base/'build-review.json')
        assert value['inventory_complete']
        core, original_data = value['observations']
        assert core['public_surface_equal'] and core['public_surface'] == core['public_surface_after']
        assert core['assembly_attributes_before'] == core['assembly_attributes_after']
        assert original_data['before_sha256'] == original_data['after_sha256'] == DATA
        inventories.append(core)
    first, second = inventories
    assert first['before_sha256'] == built['core']['sha256']
    assert first['after_sha256'] == second['before_sha256']
    assert second['after_sha256'] == CORE
    assert first['public_surface_after'] == second['public_surface']
    product = read(CONTRACTS/'build-review.json')['product']
    runtime = CONTRACTS/'build-collected/runtime'
    for name, wanted in product.items():
        assert pin(runtime/name) == wanted
        inputs[(runtime/name).relative_to(ROOT).as_posix()] = wanted
    assert product['Lokad.Onnx.dll']['sha256'] == CORE and product['Lokad.Onnx.Data.dll']['sha256'] == DATA
    return dict(passed=True, observer_rebuilt=False, product=product,
        core_public_surface_preserved=True, original_review=pin(OLD/'build-review.json'),
        original_inventory=pin(folder/'inventory/instructions.json'), methods=methods,
        consumer=built['consumer'], observed_data=built['data'], inputs=inputs)


if __name__ == '__main__':
    import json
    result = review_observer()
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','methods']}))
