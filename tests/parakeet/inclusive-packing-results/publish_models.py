"""Publish the complete eight-worker model qualification after independent audit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-inclusive-packing-models-amd-20260924'
OUTPUT = Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    proof = read(BASE / 'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(BASE / name) == wanted, name
    analysis = read(BASE / 'analysis.json'); assert pin(BASE / 'analysis.json') == proof['analysis']
    assert analysis['passed'] and analysis['no_performance_measurement'] and analysis['reference_provenance_verified']
    jobs = [f'{role}-{mode}-{isa}' for isa in ['512', '256'] for role in ['selected', 'candidate'] for mode in ['native', 'public']]
    assert list(analysis['results']) == jobs and len(analysis['resources']) == 8
    paths = [OUTPUT / f'models-{suffix}' for suffix in ['20260924.md', 'observations-20260924.json']]
    assert not any(p.exists() for p in paths)
    rows = []; inputs = {}
    for job in jobs:
        role, mode, isa = job.split('-'); result = analysis['results'][job]; assert result['passed']
        path = BASE / 'collected' / job / ('result.json' if mode == 'native' else 'output/result.json')
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
        if mode == 'native':
            report = result['native']
            assert report['numeric_gate_passed'] and report['application_passed'] and not report['failures']
            assert (report['arrays'], report['values']) == (784, 3090494)
            if role == 'candidate':
                comparisons = report['exact_selected_comparisons']
                assert len(comparisons) == 784 and all(r['bit_identical'] for r in comparisons)
            rows.append(f"| {role.title()} | {'Default' if isa == '512' else 'AVX-512 disabled'} | 784 | 3,090,494 | {report['maximum']:.12g} |")
        else:
            assert result['public_requests'] == 20
            if role == 'candidate': assert result['complete_selected_results_exact']
    samples = sum(r['samples'] for r in analysis['resources'])
    peak = max(r['peak_rss'] for r in analysis['resources'])
    lines = ['# Inclusive packing: complete Parakeet correctness', '',
        'Both products pass all native tensor and complete public-result checks in',
        'both instruction modes. Candidate tensors are bit-identical to the selected',
        'release within each mode; every complete public result also matches exactly.', '',
        '| Product | Instruction mode | Arrays | Values | Maximum scaled ORT error |',
        '|---|---|---:|---:|---:|', *rows, '',
        'The unchanged native bound is abs(actual-reference) / max(1, abs(reference))',
        '<= 1e-4. Each native worker checks the frontend, encoder, decoder outputs,',
        'decoding decisions and existing rejection/cancellation/recovery contracts.',
        'Each of four public workers checks all twenty recorded clips, complete',
        'transcripts/tokens, input immutability and retained output ownership.', '',
        'The only extra instruction setting is DOTNET_EnableAVX512=0 for the two',
        'products in the second mode. Raw settings are checked and retained exactly.',
        'The public-auditor wrapper validates that setting before neutralizing the',
        'copied flags field; every original numerical and result assertion is intact.', '',
        f'All eight workers and {samples:,} resource observations pass. Peak observed',
        f'owned RSS is {peak:,} bytes. Every owner and descendant is terminal.',
        'Compute uses AMD CPU2 and .NET 10.0.8; monitoring uses CPU0. Consumers are',
        'the retained TranscribeReplay335ca09d and AudioBenchmark7eca033a binaries.', '',
        'A read-only inspection before collection found hardlinked evidence files.',
        'The collector was frozen with one transport option, dereference=True, to',
        'export their bytes as regular archive members. Source equivalence is checked',
        'against the original collector; all manifests, safe extraction checks and',
        'hash assertions are unchanged. No inference or transfer was repeated.', '',
        'These are correctness observations, with no scored performance claim.',
        'The candidate still requires complete application, shared-model and release',
        'qualification. The selected product and BENCHMARK.md remain unchanged.', '',
        'Artifact: artifacts/parakeet-inclusive-packing-models-amd-20260924.',
        'Closure SHA256: ' + pin(BASE / 'closed.json')['sha256'] + '.',
        '[All identities, results and resource summaries](models-observations-20260924.json).', '']
    paths[0].write_text('\n'.join(lines), encoding='utf8', newline='\n')
    observations = dict(passed=True, closure=pin(BASE / 'closed.json'), analysis=pin(BASE / 'analysis.json'),
        products=analysis['identities'], consumers=analysis['consumers'], results=analysis['results'],
        resources=analysis['resources'], raw_results=inputs, collection_transport=read(BASE / 'collection-transport.json'),
        report=pin(paths[0]), generator=pin(Path(__file__)))
    paths[1].write_text(json.dumps(observations, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(dict(passed=True, files={p.name: pin(p) for p in paths})))


if __name__ == '__main__': main()
