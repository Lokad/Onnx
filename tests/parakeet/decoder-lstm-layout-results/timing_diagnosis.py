"""Describe the failed timing screen from retained clocks only; never rescore it."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-decoder-lstm-layout-timing-amd-20260927'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text())
def number(value): return value['numerator']/value['denominator']


def main():
    assert pin(BASE/'closed.json')['sha256'] == 'ac1b5e0e5400d03d84250d6148253208a6d25d3bc9e565a49718721861772fe2'
    closure = read(BASE/'closed.json')
    assert closure['passed'] and not closure['admitted']
    for name, wanted in closure['files'].items(): assert pin(BASE/name) == wanted, name
    analysis = read(BASE/'analysis.json'); performance = analysis['performance']
    assert not performance['controls_passed'] and not performance['admitted']
    assert len(performance['controls']) == 168 and len(performance['gates']) == 28
    failed = [r for r in performance['controls'] if not r['passed']]
    assert len(failed) == 100 and all(r['passed'] for r in performance['gates'])
    groups = collections.Counter((r['path'], r['kind'], 'corpus' if r['group'] == 'corpus' else 'case') for r in failed)
    processes = []
    for resource in analysis['resources'][3:]:
        name = resource['name']; value = read(BASE/'collected'/name/'output/result.json')
        rows = value['rows']; freq = value['frequency']
        phases = {}
        for phase in ['warmup', 'measured']:
            phases[phase] = [dict(repeat=i,
                seconds=sum(r['ticks'] for r in rows if r['phase'] == phase and r['repeat'] == i)/freq,
                allocated_bytes=sum(r['allocated_bytes'] for r in rows if r['phase'] == phase and r['repeat'] == i)) for i in range(5)]
        slowest = max(phases['measured'], key=lambda r: r['seconds'])['repeat']
        if 'fallback' in name: assert slowest == 1
        processes.append(dict(name=name, phases=phases, slowest_measured_repeat=slowest,
            peak_rss=resource['peak_rss'], foreign_cpu_fraction=resource['accounting']['foreign_cpu_fraction']))
    corpora = [{key: number(value) if isinstance(value, dict) and 'numerator' in value else value
        for key, value in row.items()} for row in performance['table'] if row['group'] == 'corpus']
    report = dict(screen_admitted=False, screen_rescored=False, new_inference_or_build=False,
        closure=pin(BASE/'closed.json'), candidate=analysis['identities']['candidate'],
        current=analysis['identities']['selected'], complete_call_clocks=60800,
        exact_output_arrays=182400, failed_repeatability_controls=100, total_controls=168,
        raw_performance_gates_passed=28, raw_corpus_means=corpora,
        failed_controls_by_group=[dict(path=p, kind=k, group=g, count=n) for (p,k,g),n in groups.items()],
        processes=processes, all_eight_fallback_processes_slowest_on_measured_repeat=1,
        interpretation='The raw layout signal is path-specific, but repeatability fails for both products and paths. Every fallback process slows on the second measured pass. Retained durations and allocation counts do not identify GC, JIT, scheduling or cache causes. No sample is dropped and no application trial is admitted.',
        next_observation='One diagnostic baseline-fallback process with unchanged calls, warmups and hardware policy; join call intervals with GC pauses/background work and JIT activity. No layout variant or performance score.')
    path = Path(__file__).with_name('timing-diagnosis-20260927.json')
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=pin(path), admitted=False, controls_failed=len(failed), raw_corpus_means=corpora)))


if __name__ == '__main__': main()
