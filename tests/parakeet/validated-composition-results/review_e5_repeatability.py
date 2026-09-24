"""Explain the retained e5 failure without changing or replacing any scored clock."""
import csv
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

from publish_release import ROOT, OUT, ORDER, closed, pin, read, csv_text, publish

BASE = ROOT/'artifacts/parakeet-e5-repeatability-review-20260924'


def main():
    source, proof, analysis = closed('graphs')
    assert pin(source/'closed.json')['sha256'] == '729814e9effff9c5e1456e165e7ef792bf2cba1c4978ce28ee3cb3ea7b4d1209'
    assert not proof['admitted'] and not proof['all_controls_passed']
    assert not BASE.exists()
    spec = importlib.util.spec_from_file_location('graph_statistics', ROOT/'tests/parakeet/validated-composition-graphs-amd/statistics.py')
    stats = importlib.util.module_from_spec(spec); spec.loader.exec_module(stats)
    reports = {role: read(source/'collected'/f'timing-e5-30tok-{role}'/'output/result.json') for role in ORDER}
    score = stats.summarize(reports)
    assert score == {k:v for k,v in next(r for r in analysis['performance'] if r['key']=='e5-30tok').items() if k!='key'}
    blocks = []
    for role, value in reports.items():
        for start in range(0, 780, 60):
            group = value['clocks'][start:start+60]
            seconds = [Fraction(c['ticks'], c['frequency']) for c in group]
            blocks.append(dict(process=role, first=start, last=start+59, warmup=start<600,
                mean_ms=float(sum(seconds)/60*1000), minimum_ms=float(min(seconds)*1000),
                maximum_ms=float(max(seconds)*1000)))
    assert len(blocks) == 78
    lines = ['# Retained e5 repeatability failure', '',
        'The composed Parakeet candidate remains unqualified for release. The original',
        '30-token e5 comparison is 6.769% slower on average and fails its candidate',
        'repeatability control (1.118620 > 1.10). These results remain unchanged.', '',
        '| Process | Measured mean, ms | Calls 600–659, ms | Calls 660–719, ms | Calls 720–779, ms |',
        '|---|---:|---:|---:|---:|']
    for role in ORDER:
        groups = [r for r in blocks if r['process']==role and not r['warmup']]
        lines.append('| '+role+' | '+f"{score['means'][role]*1000:.6f}"+' | '+' | '.join(f"{r['mean_ms']:.6f}" for r in groups)+' |')
    lines += ['', 'The excess in candidate-b is concentrated in its first measured block.',
        'Its largest measured calls include 620 (35.442 ms), 634 (34.944 ms) and',
        '632 (34.389 ms). This is a burst across multiple calls. Selected-b also',
        'contains a burst around calls 657–663, reaching 34.628 ms at call 659.',
        'The fixed blocks expose timing structure; they do not replace the original',
        '180-call means, justify trimming, or establish the cause.', '',
        'All six processes retain all 780 calls. All original resource, numerical,',
        'input and held-output checks passed. The resource samples have memory and',
        'affinity data, but no compilation, GC or CPU counters. Call clocks contain',
        'durations without absolute boundaries, and validation occurs between calls.',
        'These records cannot establish whether compilation, collection, scheduling',
        'or graph execution caused the bursts.', '',
        'The next diagnostic observes runtime compilation and GC events around the',
        'same 780 calls in four fresh selected/candidate/candidate/selected processes.',
        'Product binaries, graph execution, numerical checks and warmup labels remain',
        'unchanged. Instrumentation affects runtime history, so those observations',
        'cannot qualify a release or retroactively prove the original cause.', '',
        '[All 78 fixed blocks](e5-repeatability-blocks-20260924.csv),',
        '[all original clocks](graphs-clocks-20260924.csv),',
        '[original comparison and verdict](graphs-20260924.md).', '',
        'Source closure: `'+pin(source/'closed.json')['sha256']+'`.']
    BASE.mkdir()
    value = dict(passed=True, diagnostic_only=True, no_inference=True,
        source_closure=pin(source/'closed.json'), original_score=score, blocks=blocks,
        inputs={str(p.relative_to(ROOT)):pin(p) for p in [Path(__file__), ROOT/'tests/parakeet/validated-composition-graphs-amd/statistics.py']})
    (BASE/'analysis.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    documents = {'e5-repeatability-20260924.md':'\n'.join(lines)+'\n', 'e5-repeatability-blocks-20260924.csv':csv_text(blocks)}
    publish(documents)
    receipt = dict(passed=True, diagnostic_only=True, no_inference=True,
        files={'analysis.json':pin(BASE/'analysis.json')},
        reports={str((OUT/name).relative_to(ROOT)):pin(OUT/name) for name in documents})
    (BASE/'closed.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),blocks=len(blocks),score=score)))


if __name__ == '__main__': main()
