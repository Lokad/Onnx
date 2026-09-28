"""Publish a compact summary of the completed original model audit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-transpose-axis-models-amd-20260928'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    proof = json.loads((BASE/'closed.json').read_text())
    assert pin(BASE/'closed.json')['sha256'] == '8568a2299a820273b7357eaaba9c151bc83121bfc781759afbe9142b8d7b5984'
    assert proof['passed'] and proof['analysis'] == pin(BASE/'analysis.json')
    for name, wanted in proof['files'].items(): assert pin(BASE/name) == wanted, name
    analysis = json.loads((BASE/'analysis.json').read_text())
    assert analysis['passed'] and len(analysis['results']) == 8
    rows = []
    for role in ['selected', 'candidate']:
        for mode in ['512', '256']:
            native = analysis['results'][f'{role}-native-{mode}']['native']
            public = analysis['results'][f'{role}-public-{mode}']
            assert (native['arrays'], native['values']) == (784, 3090494)
            assert native['numeric_gate_passed'] and not native['failures']
            assert public['passed'] and public['public_requests'] == 20
            if role == 'candidate':
                assert len(native['exact_selected_comparisons']) == 784
                assert all(r['bit_identical'] for r in native['exact_selected_comparisons'])
                assert public['complete_selected_results_exact']
            rows.append(dict(role=role, mode=mode, arrays=784, values=3090494, public_requests=20))
    report = dict(passed=True, model_closure=pin(BASE/'closed.json'), analysis=pin(BASE/'analysis.json'),
                  identities=analysis['identities'], original_results=rows,
                  candidate_tensors_and_public_results_exact=True,
                  application_scored=False, release_admitted=False, source=pin(Path(__file__)))
    target = Path(__file__).with_name('models-20260928.json')
    with target.open('x') as stream: json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(published=pin(target), arrays=3136, values=12361976, public_requests=80)))


if __name__ == '__main__': main()
