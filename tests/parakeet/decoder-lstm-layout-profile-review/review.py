"""Prove reuse of the complete-request observer for the fixed LSTM candidate."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'owned-batch-isolation-profile-amd'
sys.path.insert(0, str(PARENT))
from reuse import review_observer, QUALIFIED_ROOT

SOURCE = ROOT/'artifacts/parakeet-decoder-lstm-layout-contracts-v3-amd-20260927'
MODELS = ROOT/'artifacts/parakeet-decoder-lstm-layout-models-amd-20260927'
OUT = ROOT/'artifacts/parakeet-decoder-lstm-layout-profile-review-20260927'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def review():
    prior = review_observer()
    assert prior['passed'] and not prior['observer_rebuilt']
    inputs = dict(prior['inputs'])
    assert pin(SOURCE/'failed.json')['sha256'] == '59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'
    failure = read(SOURCE/'failed.json')
    assert failure['terminal'] and failure['evidence_verified'] and not failure['passed']
    assert pin(MODELS/'closed.json')['sha256'] == '1acb1d884eb5293e0487e1b34d8c842b7240823040e63d6dc2a043ec89f7a111'
    model_proof = read(MODELS/'closed.json'); models = read(MODELS/'analysis.json')
    assert model_proof['passed'] and models['passed'] and model_proof['analysis'] == pin(MODELS/'analysis.json')
    path = SOURCE/'collected/inventory/instructions.json'
    assert pin(path) == failure['files']['collected/inventory/instructions.json']
    current = read(path)
    original = read(QUALIFIED_ROOT/'collected/inventory/instructions.json')
    assert current['inventory_complete'] and original['inventory_complete']
    old_core, old_data = original['observations']; core, data = current['observations']
    pair = models['identities']; product = pair['candidate']
    assert data['assembly'] == old_data['assembly'] == 'Lokad.Onnx.Data.dll'
    assert data['methods'] == data['unchanged_methods'] == 697
    assert old_data['methods'] == old_data['unchanged_methods'] == 697
    assert not data['differences'] and not data['added'] and not data['removed'] and not data['candidate_methods']
    assert data['normalized_methods'] == old_data['normalized_methods']
    assert data['method_flags_before'] == data['method_flags_after'] == old_data['method_flags_after']
    assert data['before_sha256'] == pair['selected']['Lokad.Onnx.Data.dll']['sha256']
    for old, new, name in [(old_core,core,'Lokad.Onnx.dll'), (old_data,data,'Lokad.Onnx.Data.dll')]:
        assert new['assembly'] == old['assembly'] == name
        assert old['after_sha256'] == prior['product'][name]['sha256']
        assert new['after_sha256'] == product[name]['sha256']
        assert old['public_surface_after'] == new['public_surface_after']
        assert old['assembly_attributes_after'] == new['assembly_attributes_after']
        path = MODELS/'collected/runtimes/candidate'/name
        assert pin(path) == model_proof['files'][path.relative_to(MODELS).as_posix()] == product[name]
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    for path in [Path(__file__), PARENT/'reuse.py', PARENT/'run.py', SOURCE/'failed.json',
                 SOURCE/'collected/inventory/instructions.json', MODELS/'closed.json', MODELS/'analysis.json']:
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    return dict(passed=True, candidate=product, previously_profiled_product=prior['product'],
        consumer=prior['consumer'], observed_data=prior['observed_data'],
        original_observer_review=prior['original_review'], original_observer_inventory=prior['original_inventory'],
        preserved_observer_methods=prior['methods'], data_methods_exact=697,
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
        closure = dict(passed=True, read_only=True, analysis=pin(OUT/'analysis.json'), reviewer=pin(Path(__file__)))
        save(OUT/'closed.json', closure)
        save(TOOLS/'observations-20260927.json', dict(closure=pin(OUT/'closed.json'), **value))
    else:
        assert read(OUT/'analysis.json') == value
        assert read(OUT/'closed.json')['analysis'] == pin(OUT/'analysis.json')
    print(json.dumps({k:value[k] for k in ['passed','candidate','consumer','observed_data','data_methods_exact','inference_calls']}))
