"""Publish the two fixed observations without producing a release score."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/e5-frequency-diagnostic-amd-20260925'
OUT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def spread(values):
    assert values and all(math.isfinite(v) and v>0 for v in values)
    return dict(minimum=min(values),maximum=max(values),mean=statistics.mean(values),
                max_min_ratio=max(values)/min(values),cv=statistics.pstdev(values)/statistics.mean(values))


def describe(blocks):
    wall=[b['wall_ms'] for b in blocks];frequency=[b['aperf_mperf'] for b in blocks]
    return dict(blocks=len(blocks),wall_ms=spread(wall),aperf_mperf=spread(frequency),
        frequency_adjusted_wall=spread([w*f for w,f in zip(wall,frequency,strict=True)]),
        pearson=None if len(set(wall))==1 or len(set(frequency))==1 else statistics.correlation(wall,frequency))


def main():
    assert not (OUT/'model-20260925.json').exists() and not (OUT/'model-20260925.md').exists()
    closure=read(BASE/'closed.json');assert closure['passed'] and closure['diagnostic_only'] and not closure['release_admitted']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');blocks=[];calls=[];interval_rows=[];processes={}
    assert list(analysis['reports'])==['a','b']
    for role,report in analysis['reports'].items():
        frequency=report['frequency'];assert len(frequency['blocks'])==13 and len(report['calls'])==780
        role_blocks=[dict(process=role,**b) for b in frequency['blocks']];blocks.extend(role_blocks)
        calls.extend(dict(process=role,**c) for c in report['calls'])
        processes[role]=dict(epoch=report['epoch'],markers=report['markers'],events=report['events'],
            intervals=len(frequency['intervals']),all_blocks=describe(role_blocks),
            measured_blocks=describe([b for b in role_blocks if not b['warmup']]),
            mean_measured_call_ms=statistics.mean(c['wall_ms'] for c in report['calls'][600:]))
        for row in frequency['intervals']:
            base={k:v for k,v in row.items() if k!='events'}
            for event,value in row['events'].items():interval_rows.append(dict(process=role,**base,event=event,**value))
    assert len(blocks)==26 and len(calls)==1560
    measured=describe([b for b in blocks if not b['warmup']]);assert measured['blocks']==6
    wall_ratio=measured['wall_ms']['max_min_ratio'];frequency_ratio=measured['aperf_mperf']['max_min_ratio']
    if wall_ratio<1.05:
        verdict='The measured blocks do not reproduce the prespecified 5% timing spread; the frequency hypothesis remains unresolved.'
    elif frequency_ratio<1.01:
        verdict='Measured timing varies by at least 5% while reported frequency varies by less than 1%; reported frequency does not explain the dominant variation in these observations.'
    else:
        verdict='Both timing and reported frequency vary; interpret direction and magnitude below. Correlation alone does not establish causality.'
    report=dict(passed=True,diagnostic_only=True,release_admitted=False,closure=pin(BASE/'closed.json'),
        consumer=analysis['consumer'],products=analysis['products'],resources=analysis['resources'],peak_rss=analysis['peak_rss'],
        failed_release_controls=analysis['failed_release_controls'],processes=processes,
        measured_blocks=measured,all_blocks=describe(blocks),verdict=verdict,
        minimum_coverage=min(b['coverage'] for b in blocks),
        retained_intervals=sum(p['intervals'] for p in processes.values()),
        assigned_intervals=sum(len(b['intervals']) for b in blocks),blocks=blocks)
    for name,rows in [('calls-20260925.csv',calls),('model-intervals-20260925.csv',interval_rows)]:
        assert not (OUT/name).exists()
        with (OUT/name).open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'model-20260925.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    corr=measured['pearson'];corr_text='undefined (constant values)' if corr is None else f'{corr:.6f}'
    lines=[
        '# Long-input e5 frequency observation','',f'**{verdict}**','',
        'Two identical M73 candidate processes preserve the original 600 warmup and',
        '180 measured calls each. The model, consumer, numerical checks, output ownership',
        'and runtime flags are unchanged. Only external frequency observation is added',
        'alongside the existing runtime event collector. All 1,560 calls and 3,120',
        'call markers reconcile. No release score or product change is produced.','',
        '| Process | Mean measured call (ms) | Counter epoch uncertainty (ms) | Counter intervals |',
        '|---|---:|---:|---:|']
    for role,p in processes.items():lines.append(f"| {role} | {p['mean_measured_call_ms']:.6f} | {p['epoch']['uncertainty_ns']/1e6:.6f} | {p['intervals']} |")
    lines += ['',
        'The primary comparison comprises six fixed 60-call measured blocks; all 26',
        'blocks, including warmup, remain published below. Every block mean includes',
        'every original clock. Counter sums use only complete intervals inside the',
        'block at both epoch bounds; boundary and startup intervals remain in the CSV.',
        f"Minimum coverage is {report['minimum_coverage']*100:.3f}%. All assigned frequency",
        'counts are positive with at least 99.9% reported counter running fraction.','',
        f"Measured block wall range: {measured['wall_ms']['minimum']:.6f}–{measured['wall_ms']['maximum']:.6f} ms",
        f'(maximum/minimum {wall_ratio:.6f}). APERF/MPERF range:',
        f"{measured['aperf_mperf']['minimum']:.6f}–{measured['aperf_mperf']['maximum']:.6f}",
        f'(maximum/minimum {frequency_ratio:.6f}). Pearson wall/frequency correlation:',
        f'{corr_text}. Multiplying wall time by the frequency ratio changes the',
        f"coefficient of variation from {measured['wall_ms']['cv']*100:.3f}% to",
        f"{measured['frequency_adjusted_wall']['cv']*100:.3f}%; this is descriptive, not a corrected score.",'',
        '| Process | Calls | Phase | Mean wall (ms) | APERF/MPERF | MPERF/TSC | Coverage |',
        '|---|---|---|---:|---:|---:|---:|']
    for b in blocks:
        phase='warmup' if b['warmup'] else 'measured'
        lines.append(f"| {b['process']} | {b['first']}–{b['last']} | {phase} | {b['wall_ms']:.6f} | {b['aperf_mperf']:.6f} | {b['mperf_tsc']:.6f} | {b['coverage']*100:.3f}% |")
    lines += ['',
        'APERF/MPERF describes counters exposed by this VM; it is not calibrated host',
        'frequency in MHz. CPU-wide intervals include monitoring effects and gaps',
        'between calls. Observer overhead is not subtracted. These observations cannot',
        'retroactively establish the cause of the previous failed benchmark controls.',
        'Those controls remain failed, and no warmup or admission threshold changes.','',
        f"All owners are terminal. {analysis['resources']:,} resource samples pass;",
        f"peak owned RSS is {analysis['peak_rss']:,} bytes. Counter, runtime and call",
        'outputs remain bound to the closure below.','',
        '[Every call](calls-20260925.csv), [every counter event and boundary reason](model-intervals-20260925.csv),',
        '[identities, blocks and descriptive statistics](model-20260925.json).','',
        f"Closure: `{report['closure']['sha256']}`."]
    with (OUT/'model-20260925.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(verdict=verdict,measured=measured,processes=processes)))


if __name__=='__main__':main()
