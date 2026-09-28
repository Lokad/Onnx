"""Retain the terminal first build; no successful qualification is manufactured."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-transpose-axis-build-amd-20260928'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text())


def main():
    folder = BASE/'build-collected'
    transfer = read(BASE/'build-transfer.json')
    receipt = read(folder/'build-collection.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'build-results.tar.gz')
    assert transfer['collection'] == pin(folder/'build-collection.json')
    assert receipt['terminal'] and receipt['code'] == 1
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    state = read(folder/'build-state.json')
    assert state['complete'] and state['code'] == 1 and receipt['state'] == pin(folder/'build-state.json')
    assert state['supervisor'] == read(BASE/'build-deployment.json')
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    assert [(r['name'], r['code']) for r in state['runs']] == [('sdk-version', 0), ('backend-restore', 0), ('backend-build', 1)]
    output = (folder/'logs/backend-build.stdout').read_text()
    errors = [line for line in output.splitlines() if ': error ' in line]
    assert len(errors) == 4 and all('TransposeAxisMovementTests.cs(' in line and 'error CS0198:' in line for line in errors)
    assert (folder/'logs/backend-build.stderr').read_text() == ''
    assert not (BASE/'closed.json').exists() and not (BASE/'build-review.json').exists()
    result = dict(preserved=True, passed=False, code=1, release_admitted=False, reason='Test fixture assigns readonly vector-face switch; CS0198.',
                  collection=pin(folder/'build-collection.json'), terminal_owners=receipt['identities'],
                  errors=errors, no_tests_or_inference_executed=True, recorder=pin(Path(__file__)))
    with (BASE/'failed.json').open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(preserved=True, failed=pin(BASE/'failed.json'), files=len(receipt['files']))))


if __name__ == '__main__': main()
