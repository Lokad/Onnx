"""Publish the closed ownership experiment without running or changing a model."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-weight-ownership-amd-v2-20260925'
ORIGINAL = ROOT / 'artifacts/parakeet-weight-ownership-amd-20260925'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    target = OUT / 'ownership-20260925.json'
    assert not target.exists()
    closed = read(BASE / 'closed.json')
    assert pin(BASE / 'closed.json')['sha256'] == 'abe97a41fbb60648a6ee55968cd5e36c861ac908603336bd2f465379cbb1f3cd'
    assert closed['passed'] and closed['analysis'] == pin(BASE / 'analysis.json')
    for name, wanted in closed['files'].items():
        assert pin(BASE / name) == wanted, name
    analysis = read(BASE / 'analysis.json')
    assert analysis['passed'] and analysis['diagnostic_only'] and not analysis['release_admitted']
    assert analysis['no_application_score'] and analysis['modified_graph_never_executed']
    assert analysis['prediction_passed'] == closed['prediction_passed']
    request = read(BASE / 'capture-collected/probe/request.json')
    result = read(BASE / 'capture-collected/probe/result.json')
    build = read(BASE / 'build-review.json')
    assert build['passed'] and build['zero_added_warnings'] and not build['product_rebuilt']
    assert request['passed'] and result['public_request_passed']
    assert result['held_outputs_unchanged'] and result['pcm_unchanged'] and result['contexts_live_and_shared']
    recovery = read(ROOT / 'tests/parakeet/weight-ownership-probe-v2/recovery.json')
    for key, path in {
        'original_preparation': ORIGINAL / 'prepared.json',
        'original_collection': ORIGINAL / 'build-collected/build-collection.json',
        'original_transfer': ORIGINAL / 'build-transfer.json',
        'original_state': ORIGINAL / 'build-collected/build-state.json',
        'original_source': ROOT / 'tests/parakeet/weight-ownership-probe/Program.cs.txt',
        'corrected_source': ROOT / 'tests/parakeet/weight-ownership-probe-v2/Program.cs.txt',
    }.items():
        assert pin(path) == recovery[key], key
    failure = read(ORIGINAL / 'build-collected/build-state.json')
    assert failure['complete'] and failure['code'] == 1
    assert not (ORIGINAL / 'capture-deployment.json').exists()
    cost_path = ROOT / 'tests/parakeet/feed-forward-cost-results/costs-20260925.json'
    assert pin(cost_path)['sha256'] == '233a927b373399a6c110c59a300c84799873d12e3d71eec6c099a09858d88d06'
    cost = read(cost_path)
    report = dict(
        passed=True, diagnostic_only=True, release_admitted=False, application_saving_measured=False,
        closure=pin(BASE / 'closed.json'), build_review=pin(BASE / 'build-review.json'),
        analysis=analysis, public_request=request,
        retained_packed_initializers=result['retained_packed_initializers'],
        non_target_initializers=result['non_target_initializers'],
        remaining_map_entries=result['remaining_map_entries'],
        core_sha256=result['core_sha256'], data_sha256=result['data_sha256'],
        original_failed_build=dict(code=1, error='CS0411', no_model_execution=True, evidence=recovery),
        preceding_cost_diagnosis=pin(cost_path),
        weight_packing_seconds=cost['weight_packing_seconds'],
        diagnostic_feed_forward_gap_seconds=cost['profile_gap_seconds'],
        native_release_source=cost['native_release_source'],
        inference_after_binding_removal=False, forced_gc_is_diagnostic_only=True,
        logical_representation_and_fallback_still_require_proof=True,
        publisher=pin(Path(__file__)))
    with target.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(published=pin(target), prediction_passed=analysis['prediction_passed'],
                         live_arrays=analysis['live_arrays'], managed_bytes_released=analysis['managed_bytes_released'])))


if __name__ == '__main__':
    main()
