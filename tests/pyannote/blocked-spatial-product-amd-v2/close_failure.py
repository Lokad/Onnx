"""Retain the unsupported-runtime-switch failure before a corrected protocol."""
import importlib.util
import json
from pathlib import Path
from protocol import check_sample, pin, read, save

ROOT = Path(__file__).resolve().parents[3]
FAILED = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-20260922'


def main():
    assert not (FAILED/'failure-closed.json').exists()
    prepared = read(FAILED/'prepared.json'); assert prepared['passed']
    for name, wanted in prepared['files'].items(): assert pin(ROOT/name) == wanted, name
    payload = read(FAILED/'payload/payload.json')
    assert pin(FAILED/'payload/payload.json') == prepared['payload'] and pin(FAILED/'payload.tar.gz') == prepared['archive']
    for name, wanted in payload['files'].items(): assert pin(FAILED/'payload'/name) == wanted, name
    collected = FAILED/'collected'; receipt = read(collected/'collection.json'); transfer = read(FAILED/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(FAILED/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 1 and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    state = read(collected/'identity.json'); assert state['complete'] and state['code'] == 1
    assert state['supervisor'] == read(FAILED/'deployment.json') == dict(pid=706199, birth=1790083890.0)
    row, = state['runs']; assert row['name'] == 'raw-256' and row['complete'] and row['code'] == -6
    assert not (collected/'raw-256/result.json').exists()
    stderr = (collected/'logs/raw-256.stderr').read_text()
    assert 'Unhandled exception. System.IO.InvalidDataException: Actual product instruction width' in stderr
    assert not (collected/'logs/raw-256.stdout').read_text().strip()
    assert row['preflight']['available'] >= payload['limits']['preflight_available'] and row['preflight']['tmpfs'] >= payload['limits']['preflight_tmpfs']
    samples = [json.loads(s) for s in (collected/'logs/raw-256.jsonl').read_text().splitlines()]
    assert len(samples) == row['samples'] == 5 and max(s['rss'] for s in samples) == row['peak_rss']
    for sample in samples:
        check_sample(sample)
        for member in sample['members']: assert row['members'][str(member['pid'])] == member['birth']
    spec = importlib.util.spec_from_file_location('failed_product_local_resources', ROOT/'tests/parakeet/portable-models/common.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.verify(read(FAILED/'local-inputs.json')['files'])
    local = module.resources(FAILED, 'local-controller.json', dict(restore=(8, 8, 900, False), build=(8, 8, 900, False), **{'layers-local': (12, 8, 900, True)}))
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p), birth=b) for p,b in row['members'].items()]
    analysis = dict(passed=False, retained_failure=True, failure='Unsupported DOTNET_EnableAVX512F setting; actual product width assertion failed before cases',
        numerical_cases_run=False, local_resources=local['resources'], remote_samples=len(samples), remote_peak_rss=row['peak_rss'],
        terminal_identities=receipt['identities'])
    save(FAILED/'failure-analysis.json', analysis)
    files = {p.relative_to(FAILED).as_posix(): pin(p) for p in FAILED.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(FAILED).parts)}
    save(FAILED/'failure-closed.json', dict(passed=False, retained_failure=True, files=files, local_inputs=prepared['files'],
        local_identities=local['identities'], remote_terminal=receipt['identities'], analysis=pin(FAILED/'failure-analysis.json')))
    print(json.dumps(dict(closed=pin(FAILED/'failure-closed.json'), **analysis)))


if __name__ == '__main__': main()
