"""Bind the existing parameterized Pyannote consumer to the single attention ownership pair."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT/'artifacts/parakeet-pointwise-tail-pyannote-amd-20260927'
MODELS = ROOT/'artifacts/parakeet-attention-owned-models-amd-20260928'
CONSUMER = 'd78c45b9059c0f22e55c5112ad12fe54b34d1e0567096f52d23b67d3b141a848'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def validate(previous, models, compatible, built):
    assert previous['passed'] and models['passed'] and compatible['passed'] and built['passed']
    assert previous['reference_provenance_verified'] and models['reference_provenance_verified']
    assert previous['consumers'] == {role: built['consumer'] for role in ['selected', 'candidate']}
    assert built['consumer']['sha256'] == CONSUMER
    assert compatible['qualified_model_product'] == previous['identities']['candidate']
    assert models['identities'] == dict(selected=compatible['selected'], candidate=compatible['candidate'])
    assert compatible['underlying_methods_reconciled'] == 3985
    assert compatible['original_public_bindings_preserved'] and compatible['all_original_method_flags_preserved']
    assert compatible['all_data_methods_exact'] and compatible['no_consumer_or_product_build']
    core, data = compatible['compiled_scope']
    assert core['assembly'] == 'Lokad.Onnx.dll' and core['unchanged'] == 3287
    assert core['original'] == 3288 and len(core['changed']) == 1
    assert {tuple(name.split('::')[:2]) for name in core['changed']} == {
        ('Lokad.Onnx.ComputationalGraph', 'PrepareOwnedMatMulWeights')}
    assert data == dict(assembly='Lokad.Onnx.Data.dll', original=697, unchanged=697, changed=[])
    assert compatible['focused_contracts']['sha256'] == 'a293403a5dc3a7bdabb7fdc448470feea2b54dbf139d98b7b3f9ed0eef39854f'
    assert compatible['actual_model_census']['sha256'] == 'ef4e41eca49d33f6cac4ae7d510922053874002787709cc3d83bc98c3437d3d7'
    inventory = previous['inventory']
    assert inventory['passed'] and inventory['methods'] == 96 and inventory['unchanged'] == 95
    assert inventory['assembly_hash_comparison_preserved'] and inventory['all_other_instructions_locals_branches_exceptions_exact']
    assert inventory['main_changes'] == ['argument-count', 'usage', 'expected-data-argument']
    guards = previous['identity_guards']
    assert guards['passed'] and guards['probes'] == 4 and guards['rejection_before_output']
    assert guards['consumer'] == built['consumer']
    return dict(passed=True, consumer=built['consumer'], identities=models['identities'],
        qualified_consumer_product=previous['identities']['candidate'],
        original_public_bindings_preserved=True, all_data_methods_exact=True,
        all_original_method_flags_preserved=True, underlying_methods_reconciled=3985,
        consumer_rebuilt=False, product_rebuilt=False, prior_inventory=inventory,
        prior_identity_guards=guards, fresh_identity_probes_required=True,
        fresh_complete_model_checks_required=True, release_admitted=False)


def review():
    inputs = {}; proofs = {}
    for folder, digest in [
        (PREVIOUS, 'f284cc72de08b0113daa8a5482e2a2dbea2bd6a89ea00e128a569036394eebb8'),
        (MODELS, '2d64fad91c26c00ffe7cd003aa128efc4bc92280da96a9bb805a8e2b93412177'),
    ]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); proofs[folder] = proof
        assert proof['passed'] and proof['analysis'] == pin(folder/'analysis.json')
        for name in ['closed.json', 'analysis.json']:
            path = folder/name; inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    def retained(folder, name):
        path = folder/name; wanted = pin(path)
        assert wanted == proofs[folder]['files'][name], name
        inputs[path.relative_to(ROOT).as_posix()] = wanted
        return path
    built = read(retained(PREVIOUS, 'collected/built.json'))
    compatible = read(retained(MODELS, 'collected/evidence/compatibility.json'))
    previous, models = read(PREVIOUS/'analysis.json'), read(MODELS/'analysis.json')
    result = validate(previous, models, compatible, built)
    for path in (PREVIOUS/'collected/runtimes/candidate').iterdir():
        if path.is_file(): retained(PREVIOUS, path.relative_to(PREVIOUS).as_posix())
    for name in ['GraphQualification.dll', 'GraphQualification.deps.json', 'GraphQualification.runtimeconfig.json']:
        path = retained(PREVIOUS, 'collected/built/'+name)
        assert pin(path) == pin(PREVIOUS/'collected/runtimes/candidate'/name)
    assert pin(PREVIOUS/'collected/built/GraphQualification.dll') == built['consumer']
    for name, wanted in previous['identities']['candidate'].items():
        assert pin(PREVIOUS/'collected/runtimes/candidate'/name) == wanted
    return dict(**result, inputs=inputs)


if __name__ == '__main__':
    result = review()
    print(json.dumps({k:v for k,v in result.items() if k != 'inputs'}, indent=2))
