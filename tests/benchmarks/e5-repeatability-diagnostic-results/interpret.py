"""Describe the closed observation without changing clocks or admission policy."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/e5-repeatability-diagnostic-amd-20260925'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def overlap(rows, calls):
    return sum(max(0, min(row['end_ms'], call['end_ms']) - max(row['start_ms'], call['begin_ms']))
               for row in rows for call in calls)


def main():
    proof = read(BASE/'closed.json')
    assert proof['passed'] and proof['diagnostic_only'] and not proof['release_admitted']
    assert proof['analysis'] == pin(BASE/'analysis.json')
    analysis = read(BASE/'analysis.json')
    published = read(OUT/'observations-20260925.json')
    assert published['closure'] == pin(BASE/'closed.json')
    assert published['products'] == analysis['products']
    rows = []
    for role, original in analysis['reports'].items():
        observation = published['reports'][role]
        assert observation['markers'] == original['markers'] == 1560
        assert len(original['calls']) == 780
        for before, after in zip(original['blocks'], observation['blocks'], strict=True):
            assert all(after[k] == value for k, value in before.items())
        calls = original['calls'][600:]
        assert [c['index'] for c in calls] == list(range(600, 780))
        measured_blocks = [b for b in observation['blocks'] if not b['warmup']]
        assert len(measured_blocks) == 3
        wall = statistics.mean(c['wall_ms'] for c in calls)
        cpu = statistics.mean(c['cpu_ms'] for c in calls)
        compiles = overlap(observation['compilations'], calls)
        pauses = overlap(observation['suspensions'], calls)
        late = [r for r in observation['loads'] if 600 <= r['call'] < 780]
        rows.append(dict(process=role, key=observation['key'], product=observation['product'],
            measured_calls=180, wall_ms=wall, process_cpu_ms=cpu,
            cpu_over_wall=cpu/wall,
            compilation_elapsed_overlap_ms=compiles, suspension_elapsed_overlap_ms=pauses,
            suspension_fraction=pauses/sum(c['wall_ms'] for c in calls),
            measured_product_loads=len(late),
            measured_load_tiers=dict(collections.Counter(r['tier'] for r in late)),
            measured_product_load_calls=sorted({r['call'] for r in late}),
            last_product_load_call=observation['loads'][-1]['call'],
            all_events=observation['events'], unmatched_events=len(observation['unmatched']),
            blocks=measured_blocks))
    assert sum(r['measured_calls'] for r in rows) == 1440
    assert all(r['measured_product_loads'] > 0 for r in rows if r['key'] == 'e5-8tok')
    assert all(r['measured_product_loads'] == 0 for r in rows if r['key'] == 'e5-512tok')
    result = dict(diagnostic_only=True, release_admitted=False, measurement_policy_changed=False,
        closure=pin(BASE/'closed.json'), inputs={str(p.relative_to(ROOT)): pin(p) for p in
            [Path(__file__), BASE/'analysis.json', OUT/'observations-20260925.json']}, rows=rows)
    target = OUT/'interpretation-20260925.json'
    document = OUT/'interpretation-20260925.md'
    assert not target.exists() and not document.exists()
    lines = ['# What the e5 runtime observation establishes', '',
        'The two inputs need separate explanations. These are instrumented diagnostic',
        'observations of the original products and loop; they do not replace the',
        'failed release comparison or constitute a new performance score.', '',
        '| Input / process | Product | Mean wall ms | Mean process CPU ms | Product code loads during measurement | Compilation overlap ms, total | Suspension overlap ms, total |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['key']} / {r['process']} | {r['product']} | {r['wall_ms']:.6f} | {r['process_cpu_ms']:.6f} | {r['measured_product_loads']} | {r['compilation_elapsed_overlap_ms']:.3f} | {r['suspension_elapsed_overlap_ms']:.3f} |")
    lines += ['',
        'For eight-token e5, every process loads product code after the 600-call',
        'warmup. The final such loads occur at calls 676, 732, 757 and 775.',
        'Thus a fixed call count did not give these observed processes a fully',
        'settled code state. This establishes ongoing compilation; it does not',
        'show that compilation explains the complete inter-process timing spread.', '',
        'For 512-token e5, the final product loads occur at calls 64, 63, 71 and 64.',
        'There are no product code loads in measured calls. Three processes have',
        'zero measured compilation overlap; the fourth has 31.714 ms across all',
        '180 calls. Suspension overlap is 38.847–60.007 ms across each process,',
        'less than 0.1% of measured wall duration. Those recorded pauses cannot',
        'account for the much larger observed changes between fixed blocks.', '',
        'Process CPU closely follows wall time in the long-input captures. This',
        'provides no positive evidence for substantial descheduling of the process.',
        'CPU accounting includes runtime threads and does not measure instructions,',
        'CPU cycles, cache stalls, frequency or host contention. The evidence does',
        'not identify the remaining cause. Background collection work is not bounded',
        'by suspension duration alone.', '',
        'Do not increase warmup for both inputs on this evidence. Preserve all',
        '6,240 calls, all events and both original failed controls. No timing',
        'subtraction, trimmed result, changed bound or repeated score is admitted.',
        'The next bounded investigation should inspect hardware-counter availability',
        'on the same VM, then specify one unchanged-workload observation that can',
        'distinguish instruction-count changes from execution throughput changes.',
        'Its prediction and resource limits must precede collection; it supplies',
        'diagnosis, not a replacement score. Fresh shared-model and Pyannote',
        'correctness checks can proceed independently.', '',
        'Every event stream has zero reported loss and zero unmatched runtime pairs.',
        'Compilation overlap is elapsed overlap, not CPU cost or causal attribution.',
        'Instrumentation changes runtime history and cannot establish the cause of',
        'the original uninstrumented failure retroactively.', '',
        '[Full observation and every clock](report-20260925.md),',
        '[computed summaries and input hashes](interpretation-20260925.json).', '',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    document.write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps(dict(passed=True, closure=result['closure'], processes=len(rows),
        maximum_suspension_fraction=max(r['suspension_fraction'] for r in rows if r['key']=='e5-512tok'),
        measurement_policy_changed=False)))


if __name__ == '__main__':
    main()
