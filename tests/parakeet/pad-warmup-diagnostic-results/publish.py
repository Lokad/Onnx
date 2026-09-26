"""Publish the one warmup diagnostic, preserving every fixed block and no score."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-pad-warmup-diagnostic-amd-20260926'
OUT = Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def main():
    assert pin(BASE / 'closed.json')['sha256'] == '4402da8effb9467d71fd6472678ead08b714ba147d9e3fe3f8788bb715ec9b4f'
    closed = read(BASE / 'closed.json')
    assert closed['passed'] and not closed['admitted'] and closed['original_screens_remain_rejected']
    for name, identity in closed['files'].items(): assert pin(BASE / name) == identity, name
    analysis = read(BASE / 'analysis.json')
    blocks = []; summaries = []; suffix = {}; calls = 0
    for role in ('current', 'candidate'):
        priming = read(BASE / f'collected/{role}-capture/priming.json')
        result = read(BASE / f'collected/{role}-capture/result.json')
        groups = [('prefix', group['round'], group['rows']) for group in priming['passes']]
        groups.append(('suffix', None, result['rows']))
        for phase, round_index, rows in groups:
            for row in rows:
                clocks = row['clocks']; frequency = result['frequency']
                assert len(clocks) == 780
                calls += len(clocks)
                for first in range(0, 780, 60):
                    selected = clocks[first:first + 60]
                    mean = Fraction(sum(c['ticks'] for c in selected), len(selected) * frequency)
                    blocks.append(dict(role=role, phase=phase, round=round_index, case=row['name'],
                        first=first, last=first + 59, original_warmup_label=first < 600,
                        numerator=mean.numerator, denominator=mean.denominator, seconds=float(mean),
                        median_counter_allocation_bytes=statistics.median(c['allocatedAfter'] - c['allocated'] for c in selected),
                        collection_calls=sum(any(c['after' + str(g)] != c['gc' + str(g)] for g in range(3)) for c in selected)))
                # Keep the original 600/180 cut for describing prefix evolution;
                # every prefix call remains warmup, never an admission measurement.
                mean = Fraction(sum(c['ticks'] for c in clocks[600:]), 180 * frequency)
                summaries.append(dict(role=role, phase=phase, round=round_index, case=row['name'],
                    original_last_180_mean_ms=float(mean * 1000),
                    all_780_mean_ms=sum(c['ticks'] for c in clocks) * 1000 / (780 * frequency)))
                if phase == 'suffix': suffix[role, row['name']] = mean
    comparisons = []
    for row in result['rows']:
        name = row['name']; before = suffix['current', name]; after = suffix['candidate', name]
        comparisons.append(dict(case=name, selected_ms=float(before * 1000), candidate_ms=float(after * 1000),
                                ratio=float(after / before), ratio_numerator=(after / before).numerator,
                                ratio_denominator=(after / before).denominator))
    assert calls == 93600 and len(blocks) == 1560
    observation = dict(passed=True, diagnostic_only=True, admitted=False, original_screens_remain_rejected=True,
        closure=pin(BASE / 'closed.json'), generator=pin(Path(__file__)), products=analysis['products'],
        state_condition_met=analysis['state_condition_met'], compilation_state=analysis['compilation_state'],
        priming=analysis['priming'], complete_calls=calls, complete_blocks=len(blocks),
        resources=analysis['resources'], peak_rss=analysis['peak_rss'], comparisons=comparisons,
        all_round_case_means=summaries,
        primary_reduction=1 - sum(suffix['candidate', r['name']] for r in result['rows'][:8]) /
                             sum(suffix['current', r['name']] for r in result['rows'][:8]))
    observation['primary_reduction'] = float(observation['primary_reduction'])
    with (OUT / 'blocks-20260926.csv').open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(blocks[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(blocks)
    with (OUT / 'observations-20260926.json').open('x', encoding='utf8') as stream:
        stream.write(json.dumps(observation, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(calls=calls, blocks=len(blocks), state_condition_met=analysis['state_condition_met'],
                         admitted=False, observations=pin(OUT / 'observations-20260926.json'))))


if __name__ == '__main__': main()
