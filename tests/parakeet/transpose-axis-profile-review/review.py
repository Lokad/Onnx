"""Verify retained observer compatibility with the single transpose dispatch candidate, without inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PREVIOUS = TOOLS.parent/'attention-owned-profile-review'
QUALIFIED = ROOT/'artifacts/parakeet-attention-owned-root-recovery-amd-20260928'
BUILD = ROOT/'artifacts/parakeet-transpose-axis-build-recovery-amd-20260928'
MODELS = ROOT/'artifacts/parakeet-transpose-axis-models-amd-20260928'
OUT = ROOT/'artifacts/parakeet-transpose-axis-profile-review-20260928'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def review():
    loader = importlib.util.spec_from_file_location('previous_observer_review', PREVIOUS/'review.py')
    previous = importlib.util.module_from_spec(loader); loader.loader.exec_module(previous)
    prior = previous.review()
    old_review = ROOT/'artifacts/parakeet-attention-owned-profile-review-20260928'
    assert pin(old_review/'closed.json')['sha256'] == '53225d7c463285e3cf8fd2dd4135ca4ffa30fc66df0a78d68a1df223f8918ec1'
    assert read(old_review/'analysis.json') == prior
    assert read(old_review/'closed.json')['analysis'] == pin(old_review/'analysis.json')
    assert prior['passed'] and not prior['observer_rebuilt'] and prior['inference_calls'] == 0
    assert pin(QUALIFIED/'closed.json')['sha256'] == 'ee4a38ff671cc3fd8cd608c0dc3008f5c1b99f61ba6c139f3deeaf0e9039305e'
    root_proof = read(QUALIFIED/'closed.json'); root = read(QUALIFIED/'analysis.json')
    assert root_proof['passed'] and root_proof['analysis'] == pin(QUALIFIED/'analysis.json')
    assert root['passed'] and root['measured'] == prior['candidate']
    old_path = QUALIFIED/'collected/inventory/instructions.json'
    assert pin(old_path) == root_proof['files']['collected/inventory/instructions.json']
    assert pin(BUILD/'closed.json')['sha256'] == 'c0c063bb3f73eed2d8d7372d17dd1da3ce4f6a962e190ef101b504fb245e4ecd'
    focused = read(BUILD/'closed.json'); assert focused['passed']
    assert focused['analysis'] == pin(BUILD/'analysis.json')
    assert pin(BUILD/'build-review.json')['sha256'] == 'f41cc8c3658a43c8348150f88df123691d6fdcee806a2844949aa3994c790d45'
    build = read(BUILD/'build-review.json'); assert build['passed']
    current_path = BUILD/'build-collected/logs/instructions.json'
    assert pin(current_path) == focused['files']['build-collected/logs/instructions.json']
    old = read(old_path); current = read(current_path)
    assert old['inventory_complete'] and current['inventory_complete']
    assert pin(MODELS/'closed.json')['sha256'] == '8568a2299a820273b7357eaaba9c151bc83121bfc781759afbe9142b8d7b5984'
    proof = read(MODELS/'closed.json'); models = read(MODELS/'analysis.json')
    assert proof['passed'] and models['passed'] and proof['analysis'] == pin(MODELS/'analysis.json')
    pair = models['identities']; candidate = pair['candidate']
    assert pair['selected'] == root['built'] == read(BUILD/'bundle/spec.json')['before_product']
    assert candidate == build['product']
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'c471f5d1ead5889b00b141ce34f2a7689cfe179fbf513da4ce5db84ed01ce277'
    assert pair['selected']['Lokad.Onnx.Data.dll'] == candidate['Lokad.Onnx.Data.dll']
    assert build['arithmetic_leaves_unchanged'] and build['data_binary_unchanged']
    assert build['methods'][0]['changed'] == ['Lokad.Onnx.Tensor`1[T]::TransposeInto::Void TransposeInto(Lokad.Onnx.Tensor`1[T], Lokad.Onnx.DenseTensor`1[T], Int32[])']
    old_data, data = old['observations'][1], current['observations'][1]
    assert old_data['assembly'] == data['assembly'] == 'Lokad.Onnx.Data.dll'
    assert old_data['methods'] == old_data['unchanged_methods'] == data['methods'] == data['unchanged_methods'] == 697
    assert not data['differences'] and not data['added'] and not data['removed'] and not data['candidate_methods']
    assert old_data['normalized_methods'] == data['normalized_methods']
    assert old_data['method_flags_before'] == old_data['method_flags_after'] == data['method_flags_before'] == data['method_flags_after']
    inputs = dict(prior['inputs'])
    for old_row, row, name in zip(old['observations'], current['observations'], ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll'], strict=True):
        assert old_row['assembly'] == row['assembly'] == name
        assert old_row['after_sha256'] == row['before_sha256'] == pair['selected'][name]['sha256']
        assert row['after_sha256'] == candidate[name]['sha256']
        assert old_row['public_surface_after'] == row['public_surface'] == row['public_surface_after']
        assert old_row['assembly_attributes_after'] == row['assembly_attributes_before'] == row['assembly_attributes_after']
        binary = MODELS/'collected/runtimes/candidate'/name
        assert pin(binary) == proof['files'][binary.relative_to(MODELS).as_posix()] == candidate[name]
        inputs[binary.relative_to(ROOT).as_posix()] = pin(binary)
    for path in [Path(__file__), PREVIOUS/'review.py', old_review/'closed.json', old_review/'analysis.json',
                 QUALIFIED/'closed.json', QUALIFIED/'analysis.json', old_path, BUILD/'closed.json',
                 BUILD/'build-review.json', BUILD/'bundle/spec.json', current_path, MODELS/'closed.json', MODELS/'analysis.json']:
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    return dict(passed=True, candidate=candidate, previously_profiled_product=root['built'],
        consumer=prior['consumer'], observed_data=prior['observed_data'],
        original_observer_review=prior['original_observer_review'],
        original_observer_inventory=prior['original_observer_inventory'],
        previous_compatibility=pin(old_review/'closed.json'),
        preserved_observer_methods=prior['preserved_observer_methods'], data_methods_exact=697,
        data_method_flags_equal=True, public_surfaces_and_assembly_attributes_equal=True,
        observer_rebuilt=False, inference_calls=0, fresh_profile_measured=False,
        original_failed_contracts_remain_failed=True, actual_root_qualification_required=True,
        selected_next_optimization=None, inputs=inputs)


def save(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    assert sys.argv[1:] in [[], ['--publish']]
    value = review()
    if sys.argv[1:]:
        assert not OUT.exists(); OUT.mkdir()
        save(OUT/'analysis.json', value)
        save(OUT/'closed.json', dict(passed=True, read_only=True,
            analysis=pin(OUT/'analysis.json'), reviewer=pin(Path(__file__))))
        save(TOOLS/'observations-20260928.json', dict(closure=pin(OUT/'closed.json'), **value))
    else:
        assert read(OUT/'analysis.json') == value
        assert read(OUT/'closed.json')['analysis'] == pin(OUT/'analysis.json')
    print(json.dumps({key:value[key] for key in ['passed','candidate','consumer','observed_data','data_methods_exact','inference_calls']}))
