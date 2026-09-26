"""Read retained clocks/counters; preserve every score and classify no cause as proven."""
from collections import Counter
import csv
import hashlib
import json
from math import prod
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORT = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
INPUTS = {}


def pin(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def closed(namespace, digest):
    base = ROOT / 'artifacts' / namespace
    data = (base / 'closed.json').read_bytes()
    assert pin(data)['sha256'] == digest
    INPUTS[str((base / 'closed.json').relative_to(ROOT))] = pin(data)
    receipt = json.loads(data)
    assert receipt['passed'] and not receipt['admitted']

    def read(name):
        data = (base / name).read_bytes()
        assert pin(data) == receipt['files'][name], name
        INPUTS[str((base / name).relative_to(ROOT))] = pin(data)
        return data

    return read


def stats(clocks, frequency):
    values = [c['ticks'] * 1e6 / frequency for c in clocks]
    return dict(count=len(values), mean_us=statistics.mean(values) if values else None,
                median_us=statistics.median(values) if values else None,
                max_us=max(values) if values else None)


def main():
    current = closed('parakeet-pad-current-screen-amd-20260926',
        '5b8d7df749d600238dfc6e1c9ead50486b0027754656c57d0c553573ada1b17d')
    old = closed('parakeet-pad-warmup-diagnostic-amd-20260926',
        '4402da8effb9467d71fd6472678ead08b714ba147d9e3fe3f8788bb715ec9b4f')
    blocks = []
    suffix = []
    for name in ('current-screen0-512', 'candidate-screen1-512',
                 'candidate-screen2-512', 'current-screen3-512'):
        value = json.loads(current(f'collected/{name}/result.json'))
        prefix = json.loads(current(f'collected/{name}/priming.json'))
        groups = [('prefix', g['round'], g['rows']) for g in prefix['passes']]
        groups.append(('suffix', None, value['rows']))
        for phase, index, rows in groups:
            assert len(rows) == 12
            for row in rows:
                assert len(row['clocks']) == 780
                for first in range(0, 780, 60):
                    blocks.append(dict(process=name, phase=phase, round=index,
                        case=row['name'], first=first, last=first + 59,
                        measured=phase == 'suffix' and first >= 600,
                        **stats(row['clocks'][first:first + 60], value['frequency'])))
        for row in value['rows']:
            rank = len(row['shape'])
            dims = [n + row['pads'][i] + row['pads'][rank+i]
                    for i, n in enumerate(row['shape'])]
            suffix.append(dict(process=name, case=row['name'], output_payload_bytes=4*prod(dims),
                **stats(row['clocks'][600:], value['frequency'])))
    assert len(blocks) == 3432 and sum(b['measured'] for b in blocks) == 144
    assert len(suffix) == 48

    # These are an older binary pair and instrumented workload, not counters
    # from the failed current screen. Never transfer their attribution to it.
    older = []
    for role in ('current', 'candidate'):
        value = json.loads(old(f'collected/{role}-capture/result.json'))
        for row in value['rows']:
            calls = row['clocks'][600:]
            changed = lambda c: any(c[f'after{g}'] != c[f'gc{g}'] for g in range(3))
            groups = {True: [c for c in calls if changed(c)],
                      False: [c for c in calls if not changed(c)]}
            assert sum(map(len, groups.values())) == 180
            older.append(dict(role=role, case=row['name'],
                count_changed=stats(groups[True], value['frequency']),
                count_unchanged=stats(groups[False], value['frequency']),
                median_counter_allocated_bytes=statistics.median(
                    c['allocatedAfter']-c['allocated'] for c in calls)))
    array_events = Counter()
    for line in old('collected/candidate-export/events/events.jsonl').splitlines():
        e = json.loads(line)
        if e['name'] == 'GC/AllocationTick' and e['payload']['TypeName'] == 'System.Single[]':
            p = e['payload']
            array_events[(p['AllocationKind'], int(p['ObjectSize']))] += 1

    source = {}
    for name in ('onnxruntime/core/providers/cpu/tensor/pad.cc',
                 'onnxruntime/core/framework/execution_frame.cc'):
        data = subprocess.run(['git', '-c', 'gc.auto=0', '-C', str(ROOT/'external/onnxruntime'),
            'show', ORT+':'+name], check=True, capture_output=True).stdout
        source[name] = pin(data)
    assert source['onnxruntime/core/providers/cpu/tensor/pad.cc']['sha256'] == \
        '64b7c3367a2d080fd1d88e6c5a4a66320eb1835d6b58a1b1825add4916b63052'
    for name in ('src/Lokad.Onnx/DenseTensor.cs',
                 'tests/parakeet/pad-dispatch-source/Zzz.LastAxisPadDispatch.cs'):
        INPUTS[name] = pin((ROOT/name).read_bytes())
    report = dict(diagnostic_only=True, score_changed=False, product_changed=False,
        inputs=INPUTS, generator=pin(Path(__file__).read_bytes()), ort_revision=ORT,
        ort_source=source, current_suffix=suffix, older_instrumented_suffix=older,
        older_candidate_array_allocation_events=[dict(kind=k, object_bytes=s, events=n)
            for (k,s),n in sorted(array_events.items())])
    with (OUT/'variation-observations-20260926.json').open('x', encoding='utf8') as f:
        f.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (OUT/'variation-blocks-20260926.csv').open('x', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(blocks[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(blocks)
    print(json.dumps(dict(blocks=len(blocks), suffix_rows=len(suffix), old_rows=len(older),
        large_array_events=sum(n for (k,s),n in array_events.items() if k=='Large'))))


if __name__ == '__main__':
    main()
